#!/usr/bin/env python3
"""
BETABLADE SERVER V3.4.0 (GARCH Ready): Smoothed Signal Engine
==============================================================

This version of the server produces smoother, more noise-resistant
Z-score signals by tuning the Kalman filter's EMA alphas. It remains
regime-agnostic and provides the factor data required by the new
GARCH-based client.

NEW FEATURES (V3.4.0):
1. Smoothed final Z-scores by adjusting EMA filters for cleaner signals.
"""

import asyncio
import json
import logging
import math
import random
import signal
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Protocol, Set, Tuple, Union

# Import all required dependencies with error handling
try:
    import httpx
    import numpy as np
    import websockets
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from sklearn.decomposition import IncrementalPCA

    print("✅ All dependencies loaded successfully")
except ImportError as e:
    print(f"❌ FATAL: Missing dependency: {e}")
    print("📦 Please install required packages:")
    print("pip install httpx numpy websockets scikit-learn rich")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#                                CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TradingConfig:
    """Centralized configuration for the trading engine."""
    timeframe_name: str
    pca_window_size: int
    kalman_warmup_periods: int
    min_observations: int
    downsample_factor: int = 1
    ws_url: str = "wss://api.hyperliquid.xyz/ws"
    info_url: str = "https://api.hyperliquid.xyz/info"
    pca_batch_size: int = 5
    pca_max_components: int = 2
    kalman_process_variance: float = 5e-2
    kalman_measurement_variance: float = 5e-5
    noise_threshold: float = 1e-8
    min_daily_volume: float = 100_000
    universe_update_interval: int = 300
    log_level: str = "INFO"
    dashboard_interval: float = 1.0
    ws_server_host: str = "localhost"
    ws_server_port: int = 8889
    max_ws_connections: int = 10
    max_return_clip: float = 0.05
    min_price_change_pct: float = 0.0005
    # MODIFICATION: Lowered alphas significantly for maximum smoothing
    kalman_velocity_ema_alpha: float = 0.5  # Was 0.6
    kalman_z_score_ema_alpha: float = 0.2  # Was 0.4
    signal_persistence_threshold: int = 1
    return_scaling_factor: float = 250.0
    asset_snapshot_interval: int = 10


# ... (The rest of the server code is identical to betablade_server_regime_agnostic.py)
# ═══════════════════════════════════════════════════════════════════════════
#                               CORE DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


class StrategyType(Enum):
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    UNKNOWN = "UNKNOWN"


class TimeFrame(Enum):
    SHORT = "1m"
    MEDIUM = "5m"
    LONG = "10m"


@dataclass(frozen=True)
class Signal:
    symbol: str
    strategy: StrategyType
    signal_type: SignalType
    z_score: float
    confidence: float
    signal_strength: float = 0.0
    timestamp: float = field(default_factory=time.time)
    raw_velocity: float = 0.0
    velocity_uncertainty: float = 0.0
    innovation: float = 0.0
    smoothed_z_score: float = 0.0
    persistence_count: int = 0


@dataclass
class MultiTimeframeSignal:
    symbol: str
    consensus_signal: SignalType
    short_term_signal: Signal
    medium_term_signal: Signal
    long_term_signal: Signal
    consensus_score: float
    timestamp: float = field(default_factory=time.time)
    alignment_score: float = 0.0
    stability_score: float = 0.0


@dataclass
class KalmanDebugInfo:
    symbol: str
    raw_input: float
    position: float
    velocity: float
    smoothed_velocity: float
    position_uncertainty: float
    velocity_uncertainty: float
    innovation: float
    innovation_variance: float
    kalman_gain: float
    update_count: int
    raw_z_score: float
    smoothed_z_score: float
    timestamp: float = field(default_factory=time.time)


def convert_enums_to_values(obj):
    """Convert Enum objects to their values for JSON serialization."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: convert_enums_to_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_enums_to_values(item) for item in obj]
    if hasattr(obj, '__dict__'):
        return convert_enums_to_values(obj.__dict__)
    return obj


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for trading data types."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return convert_enums_to_values(obj.__dict__)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════
#                          CORE MODELS & STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveKalmanFilter:
    """Enhanced Kalman filter with adaptive capabilities and EMA smoothing."""

    def __init__(self, process_var: float, measurement_var: float, symbol: str = "UNKNOWN",
                 velocity_ema_alpha: float = 0.7, z_score_ema_alpha: float = 0.8):
        # State transition matrix (position, velocity)
        self.F = np.array([[1., 1.], [0., 1.]])
        # Observation matrix (we observe position)
        self.H = np.array([[1., 0.]])
        # Process noise covariance
        self.Q = np.array([[1.0, 1.0], [1.0, 2.0]]) * process_var
        # Measurement noise covariance
        self.R = np.array([[measurement_var]])

        # State vector [position, velocity]
        self.x = np.zeros((2, 1))
        # Error covariance matrix
        self.P = np.eye(2) * 1.0

        self.update_count = 0
        self.symbol = symbol
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Tracking and debugging
        self.velocity_changes = deque(maxlen=10)
        self.debug_history = deque(maxlen=10)

        # EMA smoothed values
        self.smoothed_velocity = 0.0
        self.smoothed_z_score = 0.0

        # Signal persistence tracking
        self.current_signal_persistence = 0
        self.last_signal_direction = 0

        # EMA parameters
        self.velocity_ema_alpha = velocity_ema_alpha
        self.z_score_ema_alpha = z_score_ema_alpha

        # Thread safety
        self._lock = threading.RLock()

    def predict(self):
        """Prediction step of Kalman filter."""
        with self._lock:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q

            # Ensure P remains positive definite
            if np.any(np.linalg.eigvals(self.P) <= 0):
                self.P += np.eye(2) * 1e-6

    def update(self, obs: float):
        """Update step of Kalman filter with new observation."""
        if not np.isfinite(obs):
            return

        with self._lock:
            try:
                # Prediction step
                self.predict()

                # Innovation (measurement residual)
                y = obs - (self.H @ self.x)[0, 0]

                # Innovation covariance
                S = (self.H @ self.P @ self.H.T + self.R)[0, 0]

                if S <= 1e-15:
                    return

                # Kalman gain
                K = (self.P @ self.H.T) / S

                # State update
                self.x = self.x + K * y

                # Covariance update (Joseph form for numerical stability)
                I_KH = np.eye(2) - K @ self.H
                self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

                # Track velocity changes
                current_velocity = self.x[1, 0]
                self.velocity_changes.append(abs(current_velocity - self.smoothed_velocity))

                # EMA smoothing for velocity
                if self.update_count == 0:
                    self.smoothed_velocity = current_velocity
                else:
                    self.smoothed_velocity = (self.velocity_ema_alpha * current_velocity +
                                              (1 - self.velocity_ema_alpha) * self.smoothed_velocity)

                self.update_count += 1
                raw_z = self.velocity_z_score

                # EMA smoothing for z-score
                if self.update_count == 1:
                    self.smoothed_z_score = raw_z
                else:
                    self.smoothed_z_score = (self.z_score_ema_alpha * raw_z +
                                             (1 - self.z_score_ema_alpha) * self.smoothed_z_score)

                # Track signal persistence
                if abs(self.smoothed_z_score) > 0.05:
                    current_direction = 1 if self.smoothed_z_score > 0 else -1
                    if current_direction == self.last_signal_direction:
                        self.current_signal_persistence += 1
                    else:
                        self.current_signal_persistence = 1
                        self.last_signal_direction = current_direction
                else:
                    if self.current_signal_persistence > 0:
                        self.current_signal_persistence = max(0, self.current_signal_persistence - 0.5)

                # Store debug information
                debug_info = KalmanDebugInfo(
                    symbol=self.symbol,
                    raw_input=obs,
                    position=self.x[0, 0],
                    velocity=current_velocity,
                    smoothed_velocity=self.smoothed_velocity,
                    position_uncertainty=np.sqrt(max(1e-15, self.P[0, 0])),
                    velocity_uncertainty=np.sqrt(max(1e-15, self.P[1, 1])),
                    innovation=y,
                    innovation_variance=S,
                    kalman_gain=K[1, 0],
                    update_count=self.update_count,
                    raw_z_score=raw_z,
                    smoothed_z_score=self.smoothed_z_score
                )
                self.debug_history.append(debug_info)

            except Exception as e:
                self.logger.error(f"Error in Kalman filter update for {self.symbol}: {e}")

    @property
    def velocity_z_score(self) -> float:
        """Calculate velocity z-score (velocity normalized by uncertainty)."""
        with self._lock:
            if self.update_count < 3:
                return 0.0

            uncertainty = np.sqrt(max(1e-15, self.P[1, 1]))
            if uncertainty <= 1e-12:
                return 0.0

            raw_z = self.smoothed_velocity / uncertainty
            return np.clip(raw_z, -50.0, 50.0)

    @property
    def stability_score(self) -> float:
        """Calculate stability score based on velocity consistency and persistence."""
        with self._lock:
            if len(self.velocity_changes) < 2:
                return 0.8

            velocity_stability = 1.0 / (1.0 + np.std(self.velocity_changes) * 10)
            persistence_factor = min(1.0, self.current_signal_persistence / 2.0)
            return 0.5 * velocity_stability + 0.5 * persistence_factor

    def is_ready(self, min_updates: int) -> bool:
        """Check if filter has enough updates to be reliable."""
        with self._lock:
            return self.update_count >= min_updates

    def get_debug_info(self) -> Optional[KalmanDebugInfo]:
        """Get latest debug information."""
        with self._lock:
            return self.debug_history[-1] if self.debug_history else None


# ═══════════════════════════════════════════════════════════════════════════
#                           MULTI-TIMEFRAME FUSION
# ═══════════════════════════════════════════════════════════════════════════

class TimeframeManager:
    """Manages multiple timeframe signal engines and generates consensus signals."""

    def __init__(self):
        # Create signal engines for different timeframes with enhanced parameters
        self.engines = {
            TimeFrame.SHORT: SignalEngine(TradingConfig("Short_1m", 60, 5, 10, 1)),
            TimeFrame.MEDIUM: SignalEngine(TradingConfig("Medium_5m", 300, 8, 12, 3)),
            TimeFrame.LONG: SignalEngine(TradingConfig("Long_10m", 600, 10, 15, 5)),
        }

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._fused_signals_lock = threading.RLock()
        self._fused_signals: List[MultiTimeframeSignal] = []

        # Signal history tracking for graphing
        self.signal_history = defaultdict(lambda: deque(maxlen=20))
        self.consensus_ema = defaultdict(float)
        self.signal_graph_history = defaultdict(lambda: deque(maxlen=50))  # Store more points for graphing

    def generate_consensus_signals(self, symbols: List[str]):
        """Generate consensus signals across all timeframes."""
        if not symbols:
            return

        try:
            fused_signals = []
            all_signals = {}

            # Safely get signals from each engine
            for tf, engine in self.engines.items():
                try:
                    signals = engine.get_signals()
                    all_signals[tf] = {s.symbol: s for s in signals}
                except Exception as e:
                    self.logger.error(f"Error getting signals from {tf.value} engine: {e}")
                    all_signals[tf] = {}

            # Generate consensus for each symbol
            for symbol in symbols:
                try:
                    short_s = all_signals[TimeFrame.SHORT].get(symbol)
                    med_s = all_signals[TimeFrame.MEDIUM].get(symbol)
                    long_s = all_signals[TimeFrame.LONG].get(symbol)

                    if not all([short_s, med_s, long_s]):
                        continue

                    signals = [short_s, med_s, long_s]
                    alignment_score = self._calculate_alignment(signals)
                    stability_score = self._calculate_stability(symbol, signals)
                    consensus_score, consensus_signal = self._weighted_consensus(
                        symbol, signals, alignment_score, stability_score
                    )

                    fused_signal = MultiTimeframeSignal(
                        symbol=symbol,
                        consensus_signal=consensus_signal,
                        short_term_signal=short_s,
                        medium_term_signal=med_s,
                        long_term_signal=long_s,
                        consensus_score=consensus_score,
                        alignment_score=alignment_score,
                        stability_score=stability_score
                    )

                    fused_signals.append(fused_signal)
                    self.signal_history[symbol].append(consensus_score)

                    # Store signal history for graphing
                    self.signal_graph_history[symbol].append({
                        'score': consensus_score,
                        'timestamp': time.time()
                    })

                except Exception as e:
                    self.logger.error(f"Error processing consensus for {symbol}: {e}")

            with self._fused_signals_lock:
                self._fused_signals = fused_signals

        except Exception as e:
            self.logger.error(f"Error in generate_consensus_signals: {e}")

    def _calculate_alignment(self, signals: List[Signal]) -> float:
        """Calculate alignment score across timeframes."""
        if not signals:
            return 0.0

        try:
            z_scores = [s.smoothed_z_score for s in signals if np.isfinite(s.smoothed_z_score)]
            if not z_scores:
                return 0.0

            pos = sum(1 for z in z_scores if z > 0.1)
            neg = sum(1 for z in z_scores if z < -0.1)

            if (pos + neg) == 0:
                return 0.0

            return max(pos, neg) / len(z_scores)

        except Exception as e:
            self.logger.error(f"Error calculating alignment: {e}")
            return 0.0

    def _calculate_stability(self, symbol: str, signals: List[Signal]) -> float:
        """Calculate stability score based on signal history."""
        try:
            if symbol not in self.signal_history or len(self.signal_history[symbol]) < 2:
                return 0.7

            recent_scores = list(self.signal_history[symbol])
            if len(recent_scores) < 2:
                return 0.7

            score_variance = np.var(recent_scores)
            variance_stability = 1.0 / (1.0 + score_variance * 1)
            avg_signal_stability = np.mean([s.signal_strength for s in signals]) if signals else 0.7

            return 0.5 * variance_stability + 0.5 * avg_signal_stability

        except Exception as e:
            self.logger.error(f"Error calculating stability for {symbol}: {e}")
            return 0.7

    def _weighted_consensus(self, symbol: str, signals: List[Signal],
                            alignment: float, stability: float) -> Tuple[float, SignalType]:
        """Generate weighted consensus signal."""
        try:
            # Adaptive weighting based on stability
            if stability > 0.7:
                weights = [0.1, 0.2, 0.7]  # Favor long-term when stable
            elif stability > 0.4:
                weights = [0.2, 0.3, 0.5]
            else:
                weights = [0.4, 0.3, 0.3]  # Equal weight when unstable

            # Calculate weighted z-score
            weighted_z = sum(s.smoothed_z_score * w for s, w in zip(signals, weights)
                             if np.isfinite(s.smoothed_z_score))

            # Apply stability and alignment factors
            combined_factor = (0.5 + (stability * 0.5)) * (0.5 + (alignment * 0.5))
            raw_score = weighted_z * combined_factor

            # EMA smoothing of consensus score for cleaner signals
            ema_alpha = 0.3
            smoothed_score = ema_alpha * raw_score + (1 - ema_alpha) * self.consensus_ema.get(symbol, raw_score)
            self.consensus_ema[symbol] = smoothed_score

            # Determine signal type (pure momentum interpretation)
            if smoothed_score > 0.2:
                signal_type = SignalType.LONG
            elif smoothed_score < -0.2:
                signal_type = SignalType.SHORT
            else:
                signal_type = SignalType.HOLD

            return smoothed_score, signal_type

        except Exception as e:
            self.logger.error(f"Error in weighted consensus for {symbol}: {e}")
            return 0.0, SignalType.HOLD

    def get_fused_signals(self) -> List[MultiTimeframeSignal]:
        """Get current fused signals thread-safely."""
        with self._fused_signals_lock:
            return self._fused_signals.copy()


# ═══════════════════════════════════════════════════════════════════════════
#                        SINGLE TIMEFRAME SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SignalEngine:
    """Signal generation engine for a single timeframe. (Regime-Agnostic)"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.config.timeframe_name}")

        # Thread safety
        self._signals_lock = threading.RLock()
        self._filters_lock = threading.RLock()

        # Core components
        self._signals: List[Signal] = []
        self.pca = IncrementalPCA(n_components=config.pca_max_components)
        self.pca_fitted = False
        self.kalman_filters: Dict[str, AdaptiveKalmanFilter] = {}
        self.current_universe: List[str] = []
        
        self.latest_factor_returns: Optional[np.ndarray] = None


    def update_model(self, returns_matrix: np.ndarray, symbols: List[str]):
        """Update the signal generation model with new data."""
        if returns_matrix.size == 0 or not symbols:
            return

        try:
            if symbols != self.current_universe:
                self.logger.info(
                    f"{self.config.timeframe_name}: Universe changed from "
                    f"{len(self.current_universe)} to {len(symbols)} assets. Resetting PCA."
                )
                self.current_universe = symbols.copy()
                self.pca = IncrementalPCA(n_components=min(self.config.pca_max_components, len(symbols)))
                self.pca_fitted = False

            if returns_matrix.shape[1] != len(symbols):
                self.logger.error(
                    f"Matrix shape mismatch: {returns_matrix.shape[1]} columns vs {len(symbols)} symbols"
                )
                return

            self._update_pca(returns_matrix)
            if not self.pca_fitted:
                self.logger.debug(f"{self.config.timeframe_name}: PCA not fitted, skipping signal generation.")
                return
            
            residual_returns, _ = self._compute_decomposed_returns(returns_matrix)
            self._update_kalman_filters(residual_returns, symbols)
            self._generate_signals(symbols)

        except Exception as e:
            self.logger.error(f"Error in update_model for {self.config.timeframe_name}: {e}")

    def _update_pca(self, returns_matrix: np.ndarray):
        """Update PCA model with new returns data."""
        try:
            if np.any(~np.isfinite(returns_matrix)):
                returns_matrix = np.nan_to_num(returns_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            if returns_matrix.shape[0] < 2:
                return

            self.pca.partial_fit(returns_matrix)
            self.pca_fitted = True

        except Exception as e:
            self.logger.error(f"PCA update failed for {self.config.timeframe_name}: {e}")
            self.pca_fitted = False

    def _compute_decomposed_returns(self, mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose returns into systematic and residual components."""
        try:
            if not self.pca_fitted:
                return mat, np.array([])

            factors = self.pca.transform(mat)
            systematic = self.pca.inverse_transform(factors)
            residuals = mat - systematic

            residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
            factors = np.nan_to_num(factors, nan=0.0, posinf=0.0, neginf=0.0)

            self.latest_factor_returns = factors

            return residuals, factors

        except Exception as e:
            self.logger.error(f"PCA decomposition failed for {self.config.timeframe_name}: {e}")
            return mat, np.array([])

    def _update_kalman_filters(self, residual_returns: np.ndarray, symbols: List[str]):
        """Update Kalman filters with residual returns."""
        if residual_returns.shape[0] == 0:
            return

        try:
            last_residuals = residual_returns[-1, :]

            with self._filters_lock:
                for i, symbol in enumerate(symbols):
                    if i >= len(last_residuals):
                        continue

                    if symbol not in self.kalman_filters:
                        self.kalman_filters[symbol] = AdaptiveKalmanFilter(
                            self.config.kalman_process_variance,
                            self.config.kalman_measurement_variance,
                            symbol,
                            self.config.kalman_velocity_ema_alpha,
                            self.config.kalman_z_score_ema_alpha
                        )

                    if np.isfinite(last_residuals[i]):
                        self.kalman_filters[symbol].update(last_residuals[i])

        except Exception as e:
            self.logger.error(f"Error updating Kalman filters for {self.config.timeframe_name}: {e}")

    def _generate_signals(self, symbols: List[str]):
        """Generate TRADING signals based on Kalman filter outputs (Regime-Agnostic)."""
        try:
            generated_signals = []

            if "Long" in self.config.timeframe_name:
                base_threshold = 0.1
            elif "Medium" in self.config.timeframe_name:
                base_threshold = 0.12
            else:
                base_threshold = 0.15

            with self._filters_lock:
                for symbol, kf in self.kalman_filters.items():
                    if symbol not in symbols or not kf.is_ready(self.config.kalman_warmup_periods):
                        continue

                    try:
                        z = kf.smoothed_z_score

                        if not np.isfinite(z):
                            continue

                        stability = kf.stability_score
                        persistence = kf.current_signal_persistence

                        if persistence >= self.config.signal_persistence_threshold:
                            persistence_factor = min(1.0, persistence / 1.0)
                            confidence = min(abs(z) / 0.5, 1.0) * stability * persistence_factor
                            signal_strength = min(abs(z) / 0.25, 1.0) * stability * persistence_factor
                        else:
                            confidence = min(abs(z) / 1.0, 0.8) * stability
                            signal_strength = min(abs(z) / 0.5, 0.8) * stability

                        adaptive_threshold = base_threshold * (1.0 + (1.0 - stability) * 0.1)
                        sig_type = SignalType.HOLD

                        if z > adaptive_threshold:
                            sig_type = SignalType.LONG
                        elif z < -adaptive_threshold:
                            sig_type = SignalType.SHORT

                        debug_info = kf.get_debug_info()
                        if debug_info:
                            generated_signals.append(Signal(
                                symbol=symbol,
                                strategy=StrategyType.UNKNOWN,
                                signal_type=sig_type,
                                z_score=kf.velocity_z_score,
                                confidence=confidence,
                                signal_strength=signal_strength,
                                raw_velocity=debug_info.velocity,
                                velocity_uncertainty=debug_info.velocity_uncertainty,
                                innovation=debug_info.innovation,
                                smoothed_z_score=z,
                                persistence_count=int(persistence)
                            ))

                    except Exception as e:
                        self.logger.error(f"Error generating signal for {symbol}: {e}")

            with self._signals_lock:
                self._signals = generated_signals

        except Exception as e:
            self.logger.error(f"Error in _generate_signals for {self.config.timeframe_name}: {e}")

    def get_signals(self) -> List[Signal]:
        """Get current signals thread-safely."""
        with self._signals_lock:
            return self._signals.copy()

    def get_kalman_filters(self) -> Dict[str, AdaptiveKalmanFilter]:
        """Get Kalman filters thread-safely."""
        with self._filters_lock:
            return self.kalman_filters.copy()


# ═══════════════════════════════════════════════════════════════════════════
#                             MARKET DATA ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class MarketDataEngine:
    """Main market data processing and signal generation engine."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.timeframe_manager = TimeframeManager()
        self.prices = {tf: defaultdict(lambda: deque(maxlen=eng.config.pca_window_size))
                       for tf, eng in self.timeframe_manager.engines.items()}
        self.returns = {tf: defaultdict(lambda: deque(maxlen=eng.config.pca_window_size))
                        for tf, eng in self.timeframe_manager.engines.items()}
        self.asset_metadata: Dict[str, Dict] = {}
        self.trading_universe: List[str] = []
        self.last_universe_update = 0.0
        self.asset_colors: Dict[str, str] = {}
        self._color_palette: List[str] = [
            "bright_blue", "bright_cyan", "bright_green", "bright_magenta",
            "bright_red", "bright_yellow", "cyan", "green", "magenta", "red",
            "yellow", "blue", "spring_green2", "deep_pink2", "orange1"
        ]
        self.executor = ThreadPoolExecutor(max_workers=len(TimeFrame), thread_name_prefix="SignalGen")
        self._stats_lock = threading.RLock()
        self._snapshots_processed = 0
        self._last_model_update = 0.0
        self._start_time = time.time()
        self.dashboard_data = {
            "fused_signals": [],
            "engine_stats": {},
            "top_signal_debug": None
        }
        self.shutdown_event = asyncio.Event()
        self.ws_server = TradingDataWSServer(config, self)
        self.last_asset_update_time: Dict[str, float] = defaultdict(float)
        self._update_engine_stats()

    @property
    def snapshots_processed(self) -> int:
        with self._stats_lock:
            return self._snapshots_processed

    @property
    def uptime(self) -> float:
        with self._stats_lock:
            return time.time() - self._start_time

    def _increment_snapshots(self):
        with self._stats_lock:
            self._snapshots_processed += 1

    async def start(self):
        self.logger.info("🚀 Engine starting core tasks...")
        try:
            await asyncio.gather(
                self._websocket_stream(),
                self._metadata_updater(),
                self._dashboard_updater(),
                self.ws_server.start()
            )
        except asyncio.CancelledError:
            self.logger.info("Engine tasks cancelled.")
        except Exception as e:
            self.logger.error(f"Error in engine start: {e}")
        finally:
            await self.ws_server.stop()
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.logger.info("Engine has stopped.")

    async def _websocket_stream(self):
        await self._fetch_asset_metadata()
        reconnect_delay = 1
        while not self.shutdown_event.is_set():
            try:
                self.logger.info("📡 Connecting to Hyperliquid WebSocket...")
                async with websockets.connect(
                        self.config.ws_url,
                        ping_interval=20,
                        ping_timeout=10,
                        close_timeout=10
                ) as ws:
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))
                    self.logger.info("📡 WebSocket connected and subscribed to Hyperliquid.")
                    reconnect_delay = 1
                    async for msg in ws:
                        if self.shutdown_event.is_set(): break
                        try:
                            data = json.loads(msg)
                            if "data" in data and "mids" in data["data"]:
                                await self._process_market_tick(data["data"]["mids"])
                        except (json.JSONDecodeError, KeyError) as e:
                            self.logger.debug(f"Invalid message format: {e}")
                            continue
                        except Exception as e:
                            self.logger.error(f"Error processing market tick: {e}")
                            continue
            except websockets.exceptions.ConnectionClosed:
                if not self.shutdown_event.is_set():
                    self.logger.warning(f"📡 WebSocket connection closed, reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)
            except Exception as e:
                if not self.shutdown_event.is_set():
                    self.logger.warning(f"📡 WebSocket error: {e}, reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)

    async def _process_market_tick(self, mids: Dict[str, str]):
        if not mids: return
        now = time.time()
        updates_made = 0
        try:
            for symbol, price_str in mids.items():
                try:
                    if now - self.last_asset_update_time[symbol] < self.config.asset_snapshot_interval: continue
                    price = float(price_str)
                    if price > 0 and self._is_tradeable_asset(symbol):
                        if self._update_asset_data(symbol, price):
                            self.last_asset_update_time[symbol] = now
                            updates_made += 1
                except (ValueError, TypeError):
                    self.logger.debug(f"Invalid price for {symbol}: {price_str}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    continue
            if updates_made > 0:
                self._increment_snapshots()
            with self._stats_lock:
                if (now - self._last_model_update) >= self.config.pca_batch_size:
                    await self._update_signal_model()
                    self._last_model_update = now
            self._update_engine_stats()
        except Exception as e:
            self.logger.error(f"Error in _process_market_tick: {e}")

    def _update_asset_data(self, symbol: str, price: float) -> bool:
        asset_updated = False
        try:
            for tf, engine in self.timeframe_manager.engines.items():
                ph = self.prices[tf][symbol]
                if ph:
                    prev_price = ph[-1]
                    if prev_price > 0:
                        price_change_pct = abs(price - prev_price) / prev_price
                        if price_change_pct < engine.config.min_price_change_pct: continue
                        try:
                            raw_return = math.log(price / prev_price)
                            clipped = np.clip(raw_return, -engine.config.max_return_clip, engine.config.max_return_clip)
                            if abs(clipped) > engine.config.noise_threshold:
                                if "Long" in engine.config.timeframe_name:
                                    scaled_return = clipped * 500.0
                                elif "Medium" in engine.config.timeframe_name:
                                    scaled_return = clipped * 350.0
                                else:
                                    scaled_return = clipped * 250.0
                                self.returns[tf][symbol].append(scaled_return)
                                asset_updated = True
                        except (ValueError, ZeroDivisionError): continue
                ph.append(price)
        except Exception as e:
            self.logger.error(f"Error updating asset data for {symbol}: {e}")
        return asset_updated

    def _is_tradeable_asset(self, symbol: str) -> bool:
        try:
            if symbol.startswith("@"): return False
            metadata = self.asset_metadata.get(symbol, {})
            volume = float(metadata.get("dayNtlVlm", 0))
            return volume >= self.config.min_daily_volume
        except (ValueError, TypeError): return False

    async def _update_signal_model(self):
        try:
            now = time.time()
            if not self.trading_universe or (now - self.last_universe_update > self.config.universe_update_interval):
                self._update_trading_universe()
                self.last_universe_update = now
            if not self.trading_universe: return
            loop = asyncio.get_event_loop()
            tasks = []
            for tf, eng in self.timeframe_manager.engines.items():
                try:
                    returns_matrix = self._build_returns_matrix(tf, eng.config)
                    if returns_matrix.size > 0:
                        task = loop.run_in_executor(self.executor, eng.update_model, returns_matrix, self.trading_universe.copy())
                        tasks.append(task)
                except Exception as e:
                    self.logger.error(f"Error building matrix for {tf.value}: {e}")
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.timeframe_manager.generate_consensus_signals(self.trading_universe)
            self._update_dashboard_data()
            await self.ws_server.broadcast_updates()
        except Exception as e:
            self.logger.error(f"Error in _update_signal_model: {e}")

    def _update_trading_universe(self):
        try:
            cfg = self.timeframe_manager.engines[TimeFrame.SHORT].config
            now = time.time()
            candidates = {s for s, r in self.returns[TimeFrame.SHORT].items() if len(r) >= cfg.min_observations}
            active_candidates = {s for s in candidates if now - self.last_asset_update_time.get(s, 0) < (self.config.universe_update_interval * 2)}
            tradeable = sorted([s for s in active_candidates if self._is_tradeable_asset(s)])
            if len(tradeable) >= 3 and set(self.trading_universe) != set(tradeable):
                old_size = len(self.trading_universe)
                self.trading_universe = tradeable
                self.logger.info(f"✅ Trading universe updated: {old_size} -> {len(self.trading_universe)} assets.")
                for symbol in self.trading_universe:
                    if symbol not in self.asset_colors:
                        self.asset_colors[symbol] = random.choice(self._color_palette)
        except Exception as e:
            self.logger.error(f"Error updating trading universe: {e}")

    def _build_returns_matrix(self, tf: TimeFrame, cfg: TradingConfig) -> np.ndarray:
        try:
            if not self.trading_universe: return np.array([])
            all_returns = []
            for s in self.trading_universe:
                returns_data = list(self.returns[tf].get(s, []))
                if returns_data:
                    all_returns.append(np.array(returns_data))
                else:
                    all_returns.append(np.array([]))
            if not all_returns: return np.array([])
            max_len = max((len(r) for r in all_returns if len(r) > 0), default=0)
            if max_len < cfg.min_observations: return np.array([])
            ds_factor = cfg.downsample_factor
            num_rows = math.ceil(max_len / ds_factor) if ds_factor > 1 else max_len
            mat = np.full((num_rows, len(self.trading_universe)), np.nan)
            for j, returns_array in enumerate(all_returns):
                if returns_array.size > 0:
                    downsampled = returns_array[::ds_factor] if ds_factor > 1 else returns_array
                    start_idx = num_rows - len(downsampled)
                    if start_idx >= 0:
                        mat[start_idx:, j] = downsampled
                    else:
                        mat[:, j] = downsampled[-num_rows:]
            for j in range(mat.shape[1]):
                col = mat[:, j]
                if np.all(np.isnan(col)):
                    mat[:, j] = 0.0
                else:
                    col_mean = np.nanmean(col)
                    mat[np.isnan(col), j] = col_mean if np.isfinite(col_mean) else 0.0
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            return mat
        except Exception as e:
            self.logger.error(f"Error building returns matrix for {tf.value}: {e}")
            return np.array([])

    async def _metadata_updater(self):
        while not self.shutdown_event.is_set():
            try:
                await self._fetch_asset_metadata()
                await asyncio.sleep(self.config.universe_update_interval)
            except asyncio.CancelledError: break
            except Exception as e:
                self.logger.error(f"Error in metadata updater: {e}")
                await asyncio.sleep(60)

    async def _fetch_asset_metadata(self):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.post(self.config.info_url, json={"type": "metaAndAssetCtxs"})
                r.raise_for_status()
                data = r.json()
                if isinstance(data, list) and len(data) >= 2:
                    universe = data[0].get("universe", [])
                    contexts = data[1] if len(data) > 1 else []
                    if len(universe) == len(contexts):
                        self.asset_metadata = {m["name"]: {**m, **c} for m, c in zip(universe, contexts)}
                    else:
                        self.logger.warning("Metadata length mismatch, using partial data")
                        self.asset_metadata = {m["name"]: m for m in universe}
        except httpx.HTTPStatusError as e:
            self.logger.warning(f"HTTP error fetching metadata: {e.response.status_code}")
        except Exception as e:
            self.logger.warning(f"Metadata fetch failed: {e}")

    def _update_dashboard_data(self):
        try:
            fused_signals = self.timeframe_manager.get_fused_signals()
            top_signal = max(fused_signals, key=lambda s: abs(s.consensus_score), default=None)
            top_signal_debug = None
            if top_signal:
                try:
                    kf = self.timeframe_manager.engines[TimeFrame.MEDIUM].get_kalman_filters().get(top_signal.symbol)
                    if kf: top_signal_debug = kf.get_debug_info()
                except Exception as e:
                    self.logger.debug(f"Error getting debug info for top signal: {e}")
            self.dashboard_data.update({"fused_signals": fused_signals, "top_signal_debug": top_signal_debug})
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")

    def _update_engine_stats(self):
        try:
            with self._stats_lock:
                self.dashboard_data["engine_stats"] = {
                    "uptime": self.uptime,
                    "snapshots": self.snapshots_processed,
                    "universe_size": len(self.trading_universe),
                    "ws_clients": len(self.ws_server.clients),
                }
        except Exception as e:
            self.logger.error(f"Error updating engine stats: {e}")

    async def _dashboard_updater(self):
        self.logger.info("🎯 Starting live dashboard...")
        console = Console()
        try:
            with Live(self._create_dashboard_layout(), screen=True, transient=True, auto_refresh=False, console=console, refresh_per_second=2) as live:
                while not self.shutdown_event.is_set():
                    try:
                        layout = self._create_dashboard_layout()
                        live.update(layout, refresh=True)
                        await asyncio.sleep(self.config.dashboard_interval)
                    except Exception as e:
                        self.logger.error(f"Dashboard update error: {e}")
                        await asyncio.sleep(1)
        except Exception as e:
            self.logger.error(f"Dashboard failed: {e}")

    def _create_dashboard_layout(self) -> Layout:
        try:
            layout = Layout()
            layout.split(Layout(name="header", size=4), Layout(ratio=1, name="main"))
            layout["main"].split_row(Layout(name="signals", ratio=1), Layout(name="graphs", ratio=1))
            stats = self.dashboard_data.get("engine_stats", {})
            uptime_val = stats.get('uptime', 0)
            snapshots_val = stats.get('snapshots', 0)
            universe_size = stats.get('universe_size', 0)
            ws_clients = stats.get('ws_clients', 0)
            header_text = (
                f"[bold green]BETABLADE V3.4.0 - REGIME-AGNOSTIC SERVER[/]\n"
                f"⏱️  UPTIME: [bold yellow]{self._format_duration(uptime_val)}[/] | "
                f"📊 TICKS PROCESSED: [bold yellow]{snapshots_val:,}[/] | "
                f"Status: [bold cyan]RUNNING[/]\n"
                f"🔌 WS Clients: [bold white]{ws_clients}[/] | "
                f"🎯 Universe: [bold white]{universe_size} assets[/]"
            )
            layout["header"].update(Panel(header_text, box=box.DOUBLE, style="bold green"))
            top_signals = sorted(self.dashboard_data.get("fused_signals", []), key=lambda s: abs(s.consensus_score), reverse=True)
            layout["signals"].update(self._create_signal_table(top_signals[:48], self.asset_colors))
            layout["graphs"].update(self._create_combined_signal_graph(top_signals[:10], self.asset_colors))
            return layout
        except Exception as e:
            self.logger.error(f"Error creating dashboard layout: {e}")
            error_layout = Layout()
            error_layout.update(Panel(f"Dashboard Error: {e}", style="red"))
            return error_layout

    def _create_signal_table(self, signals: List[MultiTimeframeSignal], asset_colors: Dict[str, str]) -> Table:
        try:
            table = Table(box=box.ROUNDED, expand=True)
            cols = ["Symbol", "Score", "Stab", "Z-S", "Z-M", "Z-L", "Persist"]
            for col in cols: table.add_column(col, justify="center")
            for s in signals:
                try:
                    asset_color = asset_colors.get(s.symbol, "white")
                    score_text = f"[{asset_color}]{s.consensus_score:+.2f}[/]"
                    z_scores = [s.short_term_signal.smoothed_z_score, s.medium_term_signal.smoothed_z_score, s.long_term_signal.smoothed_z_score]
                    z_texts = [f"[bright_yellow]{z:+.2f}[/]" if abs(z) > 0.5 else f"{z:+.2f}" for z in z_scores]
                    p = f"{s.short_term_signal.persistence_count}/{s.medium_term_signal.persistence_count}/{s.long_term_signal.persistence_count}"
                    table.add_row(f"[{asset_color}]{s.symbol}[/]", score_text, f"{s.stability_score:.2f}", z_texts[0], z_texts[1], z_texts[2], p)
                except Exception as e:
                    self.logger.debug(f"Error adding row for {s.symbol}: {e}")
            return table
        except Exception as e:
            self.logger.error(f"Error creating signal table: {e}")
            error_table = Table(title="[red]Signal Table Error[/]")
            error_table.add_column("Error")
            error_table.add_row(str(e))
            return error_table

    def _create_combined_signal_graph(self, top_signals: List, asset_colors: Dict[str, str]) -> Panel:
        """Create a single large graph for the top signals with an X-axis."""
        if not top_signals:
            return Panel("[dim]Awaiting signal data...[/dim]", title="[bold cyan]📈 Live Signal Trends[/]", box=box.ROUNDED, expand=True)

        graph_height, graph_width = 50, 100
        
        all_scores, signal_histories = [], {}
        for signal in top_signals:
            history = list(self.timeframe_manager.signal_graph_history.get(signal.symbol, []))
            if history:
                scores = [h['score'] for h in history]
                signal_histories[signal.symbol] = scores[-graph_width:]
                all_scores.extend(scores)

        if not all_scores:
            return Panel("[dim]No historical data for top signals.[/dim]", title="[bold cyan]📈 Live Signal Trends[/]", box=box.ROUNDED, expand=True)

        display_min, display_max = -0.75, 0.75
        display_range = display_max - display_min
        if display_range < 1e-6: display_range = 1.0

        canvas = [[[] for _ in range(graph_width)] for _ in range(graph_height)]

        for symbol, scores in signal_histories.items():
            color = asset_colors.get(symbol, "white")
            for x, score in enumerate(scores):
                y = int(((score - display_min) / display_range) * (graph_height - 1))
                y = max(0, min(graph_height - 1, y))
                canvas[y][x].append(color)
        
        graph_text_rows = []
        for i in range(graph_height - 1, -1, -1):
            row_val = display_min + (i / (graph_height-1)) * display_range
            label = f"{row_val: >5.2f} ┤"
            row_content = ""
            for x in range(graph_width):
                colors_at_pixel = canvas[i][x]
                if not colors_at_pixel: row_content += " "
                elif len(colors_at_pixel) == 1: row_content += f"[{colors_at_pixel[0]}]█[/]"
                else: row_content += "[bright_yellow]*[/]"
            graph_text_rows.append(label + row_content)

        x_axis_label = " " * 6 + "└"
        x_axis_line = "─" * graph_width
        graph_text_rows.append(x_axis_label + x_axis_line)

        # MODIFICATION: Add time labels for the x-axis
        time_label_start = "<-- Older"
        time_label_end = "Newer -->"
        padding_width = graph_width - len(time_label_start) - len(time_label_end)
        x_axis_time_labels = " " * 7 + time_label_start + " " * padding_width + time_label_end
        graph_text_rows.append(x_axis_time_labels)

        final_content = Text("\n").join(Text.from_markup(row) for row in graph_text_rows)
        return Panel(final_content, title="[bold cyan]📈 Top 10 Signal Trends[/]", box=box.ROUNDED)

    def _format_duration(self, seconds: float) -> str:
        try:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m" if h > 0 else f"{m}m {s}s"
        except: return "0s"


# ═══════════════════════════════════════════════════════════════════════════
#                                WEBSOCKET SERVER
# ═══════════════════════════════════════════════════════════════════════════

class TradingDataWSServer:
    """WebSocket server for real-time data streaming."""

    def __init__(self, config: TradingConfig, engine: MarketDataEngine):
        self.config, self.engine = config, engine
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.server: Optional[asyncio.AbstractServer] = None
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self._clients_lock = threading.RLock()

    async def start(self):
        self.logger.info(f"Starting WebSocket server on {self.config.ws_server_host}:{self.config.ws_server_port}")
        try:
            self.server = await websockets.serve(self.handle_client, self.config.ws_server_host, self.config.ws_server_port, max_size=2**20, max_queue=32, compression=None)
            self.logger.info("✅ WebSocket server started successfully.")
        except Exception as e:
            self.logger.error(f"❌ WebSocket server failed to start: {e}")

    async def stop(self):
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                self.logger.info("WebSocket server stopped.")
            with self._clients_lock:
                disconnection_tasks = [self._disconnect_client(client) for client in list(self.clients)]
                if disconnection_tasks: await asyncio.gather(*disconnection_tasks, return_exceptions=True)
                self.clients.clear()
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")

    async def _disconnect_client(self, websocket):
        try: await websocket.close(code=1001)
        except Exception as e: self.logger.debug(f"Error disconnecting client: {e}")

    async def handle_client(self, websocket):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        with self._clients_lock:
            if len(self.clients) >= self.config.max_ws_connections:
                self.logger.warning(f"Max connections reached, rejecting {client_id}")
                await websocket.close(code=1013, reason="Server overloaded")
                return
            self.clients.add(websocket)
        self.logger.info(f"📡 New client connected: {client_id} ({len(self.clients)} total)")
        try:
            initial_data = await self.get_all_data()
            initial_data['type'] = 'initial'
            await websocket.send(json.dumps(initial_data, cls=CustomJSONEncoder))
            async for message in websocket:
                try:
                    request = json.loads(message)
                    await self._handle_client_request(websocket, request)
                except json.JSONDecodeError: await websocket.send(json.dumps({"error": "Invalid JSON"}, cls=CustomJSONEncoder))
                except Exception as e:
                    self.logger.error(f"Error handling client request from {client_id}: {e}")
                    await websocket.send(json.dumps({"error": str(e)}, cls=CustomJSONEncoder))
        except websockets.exceptions.ConnectionClosed: self.logger.info(f"Client {client_id} disconnected normally")
        except Exception as e: self.logger.warning(f"Client {client_id} disconnected with error: {e}")
        finally:
            with self._clients_lock:
                if websocket in self.clients: self.clients.remove(websocket)
            self.logger.info(f"Client {client_id} connection closed ({len(self.clients)} remaining)")

    async def _handle_client_request(self, websocket, request):
        command = request.get("command")
        if command == "get_all_data":
            response = await self.get_all_data()
            response['type'], response['request_id'] = 'response', request.get('request_id')
            await websocket.send(json.dumps(response, cls=CustomJSONEncoder))
        elif command == "get_debug_info":
            response = await self.get_debug_data()
            response['type'], response['request_id'] = 'debug_info_response', request.get('request_id')
            await websocket.send(json.dumps(response, cls=CustomJSONEncoder))
        elif command == "ping":
            await websocket.send(json.dumps({"type": "pong", "timestamp": time.time(), "request_id": request.get('request_id')}, cls=CustomJSONEncoder))
        else:
            await websocket.send(json.dumps({"type": "error", "message": f"Unknown command: {command}", "request_id": request.get('request_id')}, cls=CustomJSONEncoder))

    async def broadcast_updates(self):
        with self._clients_lock:
            if not self.clients: return
            clients_copy = list(self.clients)
        try:
            message = await self.get_all_data()
            message['type'] = 'broadcast'
            payload = json.dumps(message, cls=CustomJSONEncoder)
            send_tasks = [self._safe_send(client, payload) for client in clients_copy]
            if send_tasks:
                results = await asyncio.gather(*send_tasks, return_exceptions=True)
                disconnected_clients = {client for client, result in zip(clients_copy, results) if isinstance(result, Exception)}
                if disconnected_clients:
                    with self._clients_lock: self.clients.difference_update(disconnected_clients)
                    self.logger.info(f"Removed {len(disconnected_clients)} disconnected clients")
        except Exception as e:
            self.logger.error(f"Failed to broadcast updates: {e}")

    async def _safe_send(self, client, payload):
        try: await asyncio.wait_for(client.send(payload), timeout=5.0)
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError): raise
        except Exception as e:
            self.logger.debug(f"Send error to client: {e}")
            raise

    async def get_all_data(self) -> Dict[str, Any]:
        try:
            fused_signals = self.engine.timeframe_manager.get_fused_signals()
            signals_data = [asdict(s) for s in fused_signals]
            latest_prices = {s: p[-1] for s, p in self.engine.prices[TimeFrame.SHORT].items() if p}
            factor_returns_payload = None
            try:
                medium_engine = self.engine.timeframe_manager.engines.get(TimeFrame.MEDIUM)
                if medium_engine and medium_engine.latest_factor_returns is not None:
                     factor_returns_payload = medium_engine.latest_factor_returns[:, 0].tolist()
            except Exception as e:
                self.logger.error(f"Error getting factor returns for broadcast: {e}")
            return {
                "timestamp": time.time(), "signals": signals_data, "prices": latest_prices,
                "stats": self.engine.dashboard_data.get("engine_stats", {}),
                "universe": self.engine.trading_universe.copy(), "factor_returns": factor_returns_payload,
            }
        except Exception as e:
            self.logger.error(f"Error getting all data: {e}")
            return {"timestamp": time.time(), "signals": [], "prices": {}, "stats": {}, "universe": [], "factor_returns": None, "error": str(e)}

    async def get_debug_data(self) -> Dict[str, Any]:
        try:
            all_debug_info = {}
            for tf_name, engine in self.engine.timeframe_manager.engines.items():
                try:
                    filters, tf_debug = engine.get_kalman_filters(), {}
                    for symbol, kf in filters.items():
                        if symbol in self.engine.trading_universe and kf.is_ready(5):
                            debug_history = [asdict(d) for d in kf.debug_history]
                            if debug_history: tf_debug[symbol] = debug_history
                    all_debug_info[tf_name.value] = tf_debug
                except Exception as e:
                    self.logger.error(f"Error getting debug data for {tf_name.value}: {e}")
                    all_debug_info[tf_name.value] = {}
            return {"timestamp": time.time(), "kalman_debug_states": all_debug_info}
        except Exception as e:
            self.logger.error(f"Error getting debug data: {e}")
            return {"timestamp": time.time(), "kalman_debug_states": {}, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════════════
#                                       MAIN
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(config: TradingConfig):
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s|%(name)-24s|%(levelname)-8s| %(message)s", "%H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.hasHandlers(): logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    try:
        fname = f"trading_engine_V3_garch_ready_{int(time.time())}.log"
        fh = logging.FileHandler(fname, mode='w')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logging.info(f"V3.4.0 (GARCH Ready) logging initialized. Log file: {fname}")
    except Exception as e:
        logging.warning(f"Could not create log file: {e}")
    for lib in ["websockets.server", "websockets.client", "httpx", "asyncio"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

async def main():
    config = TradingConfig(timeframe_name="Base", pca_window_size=0, kalman_warmup_periods=0, min_observations=0, downsample_factor=0, log_level="INFO")
    setup_logging(config)
    logger = logging.getLogger(__name__)
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler(sig):
        if not shutdown_event.is_set():
            logger.warning(f"Received exit signal {sig.name}, initiating graceful shutdown...")
            loop.call_soon_threadsafe(shutdown_event.set)

    for sig in [signal.SIGINT, signal.SIGTERM]:
        try: loop.add_signal_handler(sig, signal_handler, sig)
        except NotImplementedError: logger.warning(f"Signal handler for {sig.name} not supported on this platform")

    engine = MarketDataEngine(config)
    engine.shutdown_event = shutdown_event
    try:
        logger.info("🚀 Starting BETABLADE V3.4.0 (GARCH Ready) trading engine...")
        engine_task = asyncio.create_task(engine.start())
        shutdown_waiter = asyncio.create_task(shutdown_event.wait())
        done, pending = await asyncio.wait({engine_task, shutdown_waiter}, return_when=asyncio.FIRST_COMPLETED)
        logger.info("🛑 Shutdown process starting...")
        for task in pending: task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        logger.info("✅ All tasks cleaned up successfully.")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
    finally:
        logger.info("🏁 Trading engine shutdown complete.")

if __name__ == "__main__":
    print("🚀 Starting BETABLADE V3.4.0 Trading Engine: GARCH-READY MODE...")
    print("📡 WebSocket Server will be available on ws://localhost:8889")
    print("\nPress Ctrl+C to stop gracefully...\n")
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Trading Engine stopped by user.")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logging.critical(f"Fatal error in main execution: {e}", exc_info=True)
        traceback.print_exc()