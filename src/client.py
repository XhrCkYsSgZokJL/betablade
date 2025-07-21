#!/usr/bin/env python3
"""
BETABLADE CLIENT V4.4 (Final V2): Enhanced UI & Calibrated GARCH
================================================================

This client features a refined UI, a live GARCH Volatility Ratio chart,
and more sensitive thresholds for regime detection.

KEY CHANGES (V4.4):
1. UI: Header layout updated to the user's specified format.
2. UI: Added a live chart of the GARCH Volatility Ratio to the dashboard.
3. CALIBRATION: GARCH regime thresholds adjusted for better sensitivity.
4. MODIFIED: Increased max positions to 20 with $5000 max per position
"""

import asyncio
import json
import logging
import math
import signal
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Deque

# Defensive imports
try:
    import numpy as np
    import websockets
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from arch import arch_model # ADDED for GARCH
except ImportError as e:
    print(f"FATAL: Missing required library: {e}")
    print("Install with: pip install numpy websockets rich arch")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
#                                CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TradingConfig:
    """Trading client configuration."""
    server_url: str = "ws://localhost:8889"
    starting_capital: float = 100_000.0
    max_positions: int = 20  # MODIFIED: Increased from 10 to 20
    max_position_size_pct: float = 0.05  # MODIFIED: Reduced from 0.15 to 0.05 (5% of capital = $5000)
    max_position_size_absolute: float = 5000.0  # ADDED: Hard cap at $5000 per position
    max_portfolio_risk_pct: float = 0.20
    min_signal_score: float = 0.2
    min_stability_score: float = 0.4
    exit_signal_threshold: float = 0.0
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    position_timeout_hours: float = 0.0833  # 5 minutes
    min_hold_time_minutes: float = 0.5  # 30 seconds
    signal_strength_threshold: float = 0.1
    dashboard_refresh_rate: float = 0.5
    reconnect_delay: float = 3.0
    log_level: str = "INFO"

# ═══════════════════════════════════════════════════════════════════════════
#                          MODELS & DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"
    TIMED_OUT = "TIMED_OUT"

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"

class TradingRegime(Enum):
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    UNKNOWN = "UNKNOWN"

class GarchVolatilityModel:
    """Regime detection using a GARCH(1,1) model on factor returns."""

    def __init__(self, history_size: int = 250):
        self.returns_history = deque(maxlen=history_size)
        self.regime = TradingRegime.UNKNOWN
        self.last_garch_fit_time = 0
        self.garch_refit_interval = 300  # Refit every 5 minutes
        self.model_fitted = False
        self.long_run_vol = 0.0
        self.latest_conditional_vol = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.RLock()
        self._last_garch_result = None  # Store the last GARCH fit result

    def update(self, factor_returns_col: np.ndarray):
        """Update with new factor returns and refit GARCH model periodically."""
        with self._lock:
            if factor_returns_col.size == 0: return

            # Server sends pre-scaled factor returns, use them directly
            old_len = len(self.returns_history)
            self.returns_history.extend(factor_returns_col)
            new_len = len(self.returns_history)
            self.logger.debug(f"Added {len(factor_returns_col)} factor returns, history: {old_len} -> {new_len}")

            if len(self.returns_history) < 100:
                self.logger.debug(f"Awaiting more data for GARCH fit: {len(self.returns_history)}/{100}")
                return

            now = time.time()
            
            # Always update the conditional volatility with new data, even if not refitting
            if self.model_fitted and len(factor_returns_col) > 0:
                try:
                    # Get the latest return to update conditional volatility
                    latest_return = factor_returns_col[-1]
                    if np.isfinite(latest_return):
                        self.logger.debug(f"Updating conditional vol with latest return: {latest_return:.6f}")
                        self._update_conditional_volatility(latest_return)
                except Exception as e:
                    self.logger.error(f"Error updating conditional volatility: {e}")
            
            # Refit the entire model periodically
            if now - self.last_garch_fit_time > self.garch_refit_interval:
                self.logger.info("Refitting GARCH model due to interval...")
                self._fit_garch()
                self.last_garch_fit_time = now

            if self.model_fitted:
                self._determine_regime()

    def _update_conditional_volatility(self, latest_return: float):
        """Update conditional volatility with new return without full refit."""
        try:
            if not hasattr(self, '_last_garch_result') or self._last_garch_result is None:
                return
                
            # Use the GARCH parameters to forecast one step ahead
            omega = self._last_garch_result.params['omega']
            alpha = self._last_garch_result.params['alpha[1]']
            beta = self._last_garch_result.params['beta[1]']
            
            # Get the last conditional variance
            last_variance = self._last_garch_result.conditional_volatility[-1] ** 2
            
            # Calculate new conditional variance: omega + alpha * (return^2) + beta * last_variance
            new_variance = omega + alpha * (latest_return ** 2) + beta * last_variance
            self.latest_conditional_vol = np.sqrt(new_variance)
            
            self.logger.debug(f"Updated conditional vol: {self.latest_conditional_vol:.4f} (was: {np.sqrt(last_variance):.4f})")
            
        except Exception as e:
            self.logger.error(f"Error in conditional volatility update: {e}")

    def _fit_garch(self):
        """Fit a GARCH(1,1) model to the historical returns."""
        try:
            if np.var(self.returns_history) < 1e-8:
                self.logger.warning("Factor return variance is near zero, skipping GARCH fit.")
                return

            # Server already sends scaled factor returns, so use them directly
            returns_data = np.array(list(self.returns_history))
            self.logger.debug(f"GARCH input data range: [{np.min(returns_data):.4f}, {np.max(returns_data):.4f}], std: {np.std(returns_data):.4f}")
            
            model = arch_model(returns_data, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off', options={'maxiter': 100})
            
            if not res.convergence_flag == 0:
                self.logger.warning(f"GARCH model did not converge: {res.convergence_flag}")
                return

            # Store the result for conditional volatility updates
            self._last_garch_result = res

            omega, alpha, beta = res.params['omega'], res.params['alpha[1]'], res.params['beta[1]']
            self.logger.debug(f"GARCH params: omega={omega:.6f}, alpha={alpha:.6f}, beta={beta:.6f}")
            
            if (1 - alpha - beta) > 1e-6:
                self.long_run_vol = np.sqrt(omega / (1 - alpha - beta))
            else:
                self.long_run_vol = np.std(returns_data)

            self.latest_conditional_vol = np.sqrt(res.conditional_volatility[-1])
            self.model_fitted = True
            self.logger.info(f"GARCH fit OK. Long-run vol: {self.long_run_vol:.4f}, Cond. vol: {self.latest_conditional_vol:.4f}")

        except Exception as e:
            self.logger.warning(f"GARCH model fit failed: {e}")
            self.model_fitted = False
            self._last_garch_result = None

    def _determine_regime(self):
        """Set the trading regime based on the GARCH volatility forecast."""
        if not self.model_fitted or self.long_run_vol < 1e-6:
            self.regime = TradingRegime.UNKNOWN
            return

        vol_ratio = self.latest_conditional_vol / self.long_run_vol
        
        # MODIFICATION: Using more standard thresholds around a baseline of 1.0.
        # If the ratio is consistently low, it may indicate a need to calibrate
        # the GARCH inputs, but these thresholds provide a sound logical basis.
        if vol_ratio > 1.1:  # Current volatility is >10% above long-run average
            self.regime = TradingRegime.MOMENTUM
        elif vol_ratio < 0.9:  # Current volatility is <10% below long-run average
            self.regime = TradingRegime.MEAN_REVERSION
        else:
            self.regime = TradingRegime.UNKNOWN
    
    @property
    def volatility_ratio(self) -> float:
        """Get the ratio of current conditional vol to long-run vol."""
        with self._lock:
            if not self.model_fitted or self.long_run_vol < 1e-6:
                return 1.0
            return self.latest_conditional_vol / self.long_run_vol


@dataclass
class Position:
    symbol: str; side: PositionSide; size: float; entry_price: float; entry_time: float
    stop_loss_price: float; take_profit_price: float; signal_score: float; stability_score: float
    regime: str; current_price: float = 0.0; unrealized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.OPEN; close_time: Optional[float] = None
    close_reason: str = ""; position_id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))

    def update_price(self, price: float):
        if price <= 0: return
        self.current_price = price
        shares = self.size / self.entry_price
        pnl_direction = 1 if self.side == PositionSide.LONG else -1
        self.unrealized_pnl = (price - self.entry_price) * shares * pnl_direction

    def should_stop_loss(self) -> bool:
        return (self.side == PositionSide.LONG and self.current_price <= self.stop_loss_price) or \
               (self.side == PositionSide.SHORT and self.current_price >= self.stop_loss_price)

    def should_take_profit(self) -> bool:
        return (self.side == PositionSide.LONG and self.current_price >= self.take_profit_price) or \
               (self.side == PositionSide.SHORT and self.current_price <= self.take_profit_price)

    def is_timed_out(self, timeout_hours: float) -> bool: return (time.time() - self.entry_time) > (timeout_hours * 3600)
    def get_duration_minutes(self) -> float: return ((self.close_time or time.time()) - self.entry_time) / 60.0
    def get_return_pct(self) -> float: return (self.unrealized_pnl / self.size) * 100 if self.size > 0 else 0.0
    def is_min_hold_time_met(self, min_minutes: float) -> bool: return self.get_duration_minutes() >= min_minutes

@dataclass
class TradingMetrics:
    total_pnl: float = 0.0; realized_pnl: float = 0.0; unrealized_pnl: float = 0.0; total_fees: float = 0.0
    net_pnl: float = 0.0; total_trades: int = 0; winning_trades: int = 0; losing_trades: int = 0
    win_rate: float = 0.0; avg_win: float = 0.0; avg_loss: float = 0.0; max_drawdown: float = 0.0
    profit_factor: float = 0.0; mean_reversion_trades: int = 0; momentum_trades: int = 0
    mean_reversion_pnl: float = 0.0; momentum_pnl: float = 0.0

# ═══════════════════════════════════════════════════════════════════════════
#                                TRADING CLIENT
# ═══════════════════════════════════════════════════════════════════════════

class BetaBladeClient:
    """Client with GARCH model and new manual regime override."""
    
    def __init__(self, config: TradingConfig, loop: asyncio.AbstractEventLoop):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.loop = loop
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Position] = []
        self.capital = config.starting_capital
        self.initial_capital = config.starting_capital
        self.metrics = TradingMetrics()
        self.prices: Dict[str, float] = {}
        self.signals: List[Dict] = []
        self.server_stats: Dict = {}
        self.volatility_model = GarchVolatilityModel()
        self.active_regime: str = TradingRegime.UNKNOWN.value
        self.regime_confidence: float = 0.0
        self.manual_override = False
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.last_data_time = 0.0
        self.shutdown_event = asyncio.Event()
        self.data_lock = threading.RLock()
        self.equity_peak = config.starting_capital
        self.start_time = time.time()
        self.regime_history: Deque[str] = deque(maxlen=50)
        self.regime_change_time = time.time()
        # MODIFICATION: Added history for the new chart
        self.vol_ratio_history: Deque[float] = deque(maxlen=120)

    async def set_manual_regime(self, key: str):
        with self.data_lock:
            key = key.lower()
            new_regime = self.active_regime
            if key == 'a':
                self.manual_override = False
                self.logger.warning(" MANUAL OVERRIDE DISABLED. GARCH model is now in control.")
                return
            elif key == 'm': new_regime = TradingRegime.MOMENTUM.value
            elif key == 'r': new_regime = TradingRegime.MEAN_REVERSION.value
            elif key == 'u': new_regime = TradingRegime.UNKNOWN.value
            else: return

            if self.active_regime != new_regime or not self.manual_override:
                self.manual_override = True
                self.logger.warning(f" MANUAL OVERRIDE ACTIVATED. Forcing regime to {new_regime}.")
                await self._handle_regime_change(self.active_regime, new_regime)
                self.active_regime = new_regime
                
    def _keyboard_input_loop(self):
        time.sleep(3)
        while not self.shutdown_event.is_set():
            try:
                print("\nEnter command (M, R, U, A): ", end="", flush=True)
                key = input()
                if key:
                    future = asyncio.run_coroutine_threadsafe(self.set_manual_regime(key), self.loop)
                    future.result()
            except (EOFError, KeyboardInterrupt):
                self.logger.info("Keyboard input thread stopping.")
                break
            except Exception as e:
                self.logger.error(f"Error in keyboard input loop: {e}")

    def _interpret_signal_for_regime(self, consensus_score: float, regime: str) -> tuple[str, float]:
        signal_strength = abs(consensus_score)
        if regime == TradingRegime.MOMENTUM.value:
            if consensus_score > 0: return "LONG", signal_strength
            if consensus_score < 0: return "SHORT", signal_strength
        elif regime == TradingRegime.MEAN_REVERSION.value:
            if consensus_score > 0: return "SHORT", signal_strength
            if consensus_score < 0: return "LONG", signal_strength
        return "HOLD", 0.0

    async def start(self):
        self.logger.info("🚀 Starting BETABLADE Trading Client V4.4...")
        try:
            await asyncio.gather(self._websocket_handler(), self._trading_loop(), self._dashboard_loop())
        except asyncio.CancelledError:
            self.logger.info("Trading client cancelled.")
        finally:
            await self._cleanup()

    async def _websocket_handler(self):
        while not self.shutdown_event.is_set():
            try:
                self.logger.info(f"🔌 Connecting to BETABLADE server at {self.config.server_url}")
                async with websockets.connect(self.config.server_url, ping_interval=20, ping_timeout=10, close_timeout=10) as ws:
                    self.websocket, self.is_connected = ws, True
                    self.logger.info("✅ Connected to BETABLADE server!")
                    await ws.send(json.dumps({"command": "get_all_data"}))
                    async for message in ws:
                        if self.shutdown_event.is_set(): break
                        await self._handle_server_message(message)
            except websockets.exceptions.ConnectionClosed:
                if not self.shutdown_event.is_set(): self.logger.warning("📡 Connection lost, reconnecting...")
            except Exception as e:
                if not self.shutdown_event.is_set(): self.logger.error(f"WebSocket error: {e}")
            finally:
                self.is_connected = False
                if not self.shutdown_event.is_set(): await asyncio.sleep(self.config.reconnect_delay)

    async def _handle_server_message(self, message: str):
        try:
            data = json.loads(message)
            with self.data_lock:
                if "prices" in data: self.prices.update(data.get("prices", {}))
                if "signals" in data: self.signals = data.get("signals", [])
                if "stats" in data: self.server_stats = data.get("stats", {})
                
                if not self.manual_override and "factor_returns" in data and data["factor_returns"]:
                    try:
                        factor_returns_col = np.array(data["factor_returns"])
                        self.logger.debug(f"Received {len(factor_returns_col)} factor returns from server: {factor_returns_col[:5]}...")
                        self.volatility_model.update(factor_returns_col)
                        new_regime = self.volatility_model.regime.value
                        self.regime_confidence = self.volatility_model.volatility_ratio
                        self.logger.debug(f"GARCH Vol Ratio: {self.regime_confidence:.4f}")
                        self.vol_ratio_history.append(self.regime_confidence) # Store for chart
                        
                        if self.active_regime != new_regime:
                            old_regime = self.active_regime
                            self.active_regime = new_regime
                            self.regime_change_time = time.time()
                            self.regime_history.append(self.active_regime)
                            self.logger.info(f"🔄 GARCH REGIME CHANGE: {old_regime} → {self.active_regime} (Vol Ratio: {self.regime_confidence:.2f})")
                            await self._handle_regime_change(old_regime, new_regime)
                    except Exception as e:
                        self.logger.error(f"Error processing factor returns: {e}")
                self.last_data_time = time.time()
        except Exception as e:
            self.logger.error(f"Error handling server message: {e}")

    async def _handle_regime_change(self, old_regime: str, new_regime: str):
        if new_regime == TradingRegime.UNKNOWN.value and self.positions:
            self.logger.warning("Entering UNKNOWN regime. Closing all positions for safety.")
            for symbol in list(self.positions.keys()): await self._close_position(symbol, "REGIME_UNKNOWN")
            return
        
        positions_to_evaluate = [p for p in self.positions.values() if p.regime != new_regime]
        if positions_to_evaluate:
            self.logger.info(f"Exiting {len(positions_to_evaluate)} positions due to regime change from {old_regime} to {new_regime}.")
            for pos in positions_to_evaluate: await self._close_position(pos.symbol, f"REGIME_CHANGE")

    async def _trading_loop(self):
        while not self.shutdown_event.is_set():
            if self.is_connected:
                try:
                    await self._update_positions()
                    await self._process_signals()
                    self._update_metrics()
                except Exception as e: self.logger.error(f"Error in trading loop: {e}", exc_info=False)
            await asyncio.sleep(2)

    async def _update_positions(self):
        with self.data_lock:
            prices, signals, current_regime = self.prices.copy(), self.signals.copy(), self.active_regime
        signal_lookup = {s.get("symbol"): s for s in signals if s.get("symbol")}
        
        for pos in list(self.positions.values()):
            if pos.symbol not in prices: continue
            pos.update_price(prices[pos.symbol])
            exit_reason = None
            if pos.should_stop_loss(): exit_reason = "STOP_LOSS"
            elif pos.should_take_profit(): exit_reason = "TAKE_PROFIT"
            elif pos.is_timed_out(self.config.position_timeout_hours): exit_reason = "TIMEOUT"
            elif pos.is_min_hold_time_met(self.config.min_hold_time_minutes) and \
                 self._should_exit_on_signal(pos, signal_lookup.get(pos.symbol), current_regime):
                exit_reason = "SIGNAL_EXIT"
            if exit_reason: await self._close_position(pos.symbol, exit_reason)

    async def _process_signals(self):
        with self.data_lock:
            signals, prices, current_regime = self.signals.copy(), self.prices.copy(), self.active_regime
        if current_regime == TradingRegime.UNKNOWN.value or len(self.positions) >= self.config.max_positions: return

        actionable_signals = []
        for s_data in signals:
            symbol, score, stability = s_data.get("symbol"), s_data.get("consensus_score", 0), s_data.get("stability_score", 0)
            if symbol and symbol not in self.positions and abs(score) >= self.config.min_signal_score and \
               stability >= self.config.min_stability_score and symbol in prices:
                interpreted_signal, adj_score = self._interpret_signal_for_regime(score, current_regime)
                if interpreted_signal != "HOLD" and adj_score >= self.config.signal_strength_threshold:
                    actionable_signals.append({'symbol': symbol, 'type': interpreted_signal, 'score': adj_score, 
                                               'stability': stability, 'price': prices[symbol], 'regime': current_regime})
        
        actionable_signals.sort(key=lambda x: x['score'] * x['stability'], reverse=True)
        # MODIFIED: Increased max simultaneous positions from 3 to 5, allowing more diversification
        max_new_positions = min(5, self.config.max_positions - len(self.positions))
        for signal in actionable_signals[:max_new_positions]:
            await self._open_position(signal)

    async def _open_position(self, signal: Dict):
        symbol, side_str, price, score, stability, regime = signal['symbol'], signal['type'], signal['price'], signal['score'], signal['stability'], signal['regime']
        size = self._calculate_position_size(price, score, stability)
        if size <= 100: return
        side = PositionSide.LONG if side_str == "LONG" else PositionSide.SHORT
        sl_pct, tp_pct = self.config.stop_loss_pct, self.config.take_profit_pct
        sl_price = price * (1 - sl_pct) if side == PositionSide.LONG else price * (1 + sl_pct)
        tp_price = price * (1 + tp_pct) if side == PositionSide.LONG else price * (1 - tp_pct)
        pos = Position(symbol=symbol, side=side, size=size, entry_price=price, entry_time=time.time(), stop_loss_price=sl_price,
                       take_profit_price=tp_price, signal_score=score, stability_score=stability, regime=regime, current_price=price)
        self.positions[symbol] = pos
        fee = size * 0.0001
        self.capital -= fee
        self.metrics.total_fees += fee
        regime_emoji = '📈' if regime == TradingRegime.MOMENTUM.value else '↩️'
        side_emoji = '🟢' if side == PositionSide.LONG else '🔴'
        self.logger.info(f"{side_emoji}{regime_emoji} OPEN {side.value} {symbol} @ ${price:.4f} | Size: ${size:,.0f} in {regime} regime")

    async def _close_position(self, symbol: str, reason: str):
        if symbol not in self.positions: return
        pos = self.positions.pop(symbol)
        pos.close_time, pos.close_reason = time.time(), reason
        pos.status = PositionStatus.STOPPED if reason == "STOP_LOSS" else PositionStatus.TIMED_OUT if reason == "TIMEOUT" else PositionStatus.CLOSED
        fee = pos.size * 0.0001
        self.capital += pos.unrealized_pnl - fee
        self.metrics.total_fees += fee
        if pos.regime == TradingRegime.MEAN_REVERSION.value:
            self.metrics.mean_reversion_trades += 1; self.metrics.mean_reversion_pnl += pos.unrealized_pnl
        elif pos.regime == TradingRegime.MOMENTUM.value:
            self.metrics.momentum_trades += 1; self.metrics.momentum_pnl += pos.unrealized_pnl
        self.position_history.append(pos)
        self.logger.info(f"{'💰' if pos.unrealized_pnl > 0 else '💸'} CLOSED {pos.side.value} {symbol} @ ${pos.current_price:.4f}, P&L: ${pos.unrealized_pnl:+.2f}, Reason: {reason}")

    def _calculate_position_size(self, price: float, strength: float, stability: float) -> float:
        if price <= 0: return 0
        quality_score = strength * stability
        if quality_score >= 0.6: quality_multiplier = 1.0
        elif quality_score >= 0.4: quality_multiplier = 0.8
        else: quality_multiplier = 0.5
        
        # MODIFIED: Enhanced position sizing logic with absolute cap
        size_pct = self.config.max_position_size_pct * quality_multiplier
        risk_per_trade = self.capital * (self.config.max_portfolio_risk_pct / self.config.max_positions)
        
        # Calculate size based on percentage of capital
        percentage_based_size = self.capital * size_pct
        
        # Calculate size based on risk management
        risk_based_size = risk_per_trade / self.config.stop_loss_pct
        
        # Take the minimum of all three constraints: percentage, risk-based, and absolute cap
        calculated_size = min(percentage_based_size, risk_based_size, self.config.max_position_size_absolute)
        
        self.logger.debug(f"Position sizing for {price:.4f}: pct_size=${percentage_based_size:.0f}, risk_size=${risk_based_size:.0f}, cap=${self.config.max_position_size_absolute:.0f}, final=${calculated_size:.0f}")
        
        return calculated_size

    def _should_exit_on_signal(self, pos: Position, signal: Optional[Dict], regime: str) -> bool:
        if not signal: return True
        score = signal.get("consensus_score", 0)
        interpreted_signal, adj_score = self._interpret_signal_for_regime(score, regime)
        if adj_score < self.config.exit_signal_threshold: return True
        if (pos.side == PositionSide.LONG and interpreted_signal == "SHORT") or \
           (pos.side == PositionSide.SHORT and interpreted_signal == "LONG"): return True
        return False

    def _update_metrics(self):
        open_pnl = sum(p.unrealized_pnl for p in self.positions.values())
        self.metrics.realized_pnl = sum(p.unrealized_pnl for p in self.position_history)
        self.metrics.unrealized_pnl = open_pnl
        self.metrics.total_pnl = self.metrics.realized_pnl + open_pnl
        self.metrics.net_pnl = self.metrics.total_pnl - self.metrics.total_fees
        self.metrics.total_trades = len(self.position_history)
        wins = [p for p in self.position_history if p.unrealized_pnl > 0]
        losses = [p for p in self.position_history if p.unrealized_pnl < 0]
        self.metrics.winning_trades, self.metrics.losing_trades = len(wins), len(losses)
        self.metrics.win_rate = (len(wins) / self.metrics.total_trades * 100) if self.metrics.total_trades else 0
        self.metrics.avg_win = np.mean([p.unrealized_pnl for p in wins]) if wins else 0
        self.metrics.avg_loss = np.mean([p.unrealized_pnl for p in losses]) if losses else 0
        total_wins = sum(p.unrealized_pnl for p in wins)
        total_losses = abs(sum(p.unrealized_pnl for p in losses))
        self.metrics.profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        current_equity = self.capital + open_pnl
        if current_equity > self.equity_peak: self.equity_peak = current_equity
        drawdown = (self.equity_peak - current_equity) / self.equity_peak * 100 if self.equity_peak > 0 else 0
        self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)

    async def _dashboard_loop(self):
        console = Console()
        try:
            with Live(self._create_dashboard(), console=console, refresh_per_second=2, screen=True, transient=True) as live:
                while not self.shutdown_event.is_set():
                    live.update(self._create_dashboard())
                    await asyncio.sleep(self.config.dashboard_refresh_rate)
        except Exception as e: self.logger.error(f"Dashboard error: {e}", exc_info=False)

    def _create_dashboard(self) -> Layout:
        layout = Layout()
        left_column = Layout(name="left", ratio=2)
        
        # Define the layout for the left side of the screen - split 50/50
        left_column.split(
            Layout(self._create_vol_ratio_chart(), name="vol_chart", ratio=1),
            Layout(self._create_positions_table(), name="positions", ratio=1)
        )

        layout.split(
            Layout(self._create_header(), name="header", size=4),
            Layout(ratio=1, name="main"),
            Layout(self._create_footer(), name="footer", size=1)
        )
        # Main layout splits into the new left column and the existing right (metrics)
        layout["main"].split_row(
            left_column,
            Layout(self._create_metrics_panel(), name="metrics", ratio=1)
        )
        return layout

    def _create_header(self) -> Panel:
        status = 'RUNNING' if self.is_connected else 'CONNECTING...'
        uptime_seconds = time.time() - self.start_time
        uptime_str = time.strftime("%Mm %Ss", time.gmtime(uptime_seconds))
        ticks = self.server_stats.get('snapshots', 0)
        
        title = "[bold cyan]💎 BETABLADE V4.4 - REGIME-AGNOSTIC CLIENT (20 Pos Max)[/]"
        line1 = f"⏱️  UPTIME: [yellow]{uptime_str}[/] 📊 TICKS PROCESSED: [yellow]{ticks:,}[/] | Status: [bold green]{status}[/]"

        header_text = f"{title}\n{line1}"
        return Panel(header_text, box=box.DOUBLE, style="bold blue")

    def _create_metrics_panel(self) -> Panel:
        m = self.metrics
        eq = self.capital + m.unrealized_pnl
        ret_pct = (eq - self.initial_capital) / self.initial_capital * 100 if self.initial_capital > 0 else 0
        pnl_color = "green" if m.net_pnl >= 0 else "red"
        
        regime_map = {TradingRegime.MOMENTUM.value: ("📈", "yellow"), TradingRegime.MEAN_REVERSION.value: ("↩️", "cyan"), TradingRegime.UNKNOWN.value: ("❓", "red")}
        regime_emoji, regime_color = regime_map.get(self.active_regime, ("❓", "white"))
        regime_mode_str = "[bold red]MANUAL[/]" if self.manual_override else "[bold green]AUTO[/]"

        mr_trades, mom_trades = m.mean_reversion_trades, m.momentum_trades
        mr_wins = len([p for p in self.position_history if p.regime == TradingRegime.MEAN_REVERSION.value and p.unrealized_pnl > 0])
        mom_wins = len([p for p in self.position_history if p.regime == TradingRegime.MOMENTUM.value and p.unrealized_pnl > 0])
        mr_wr = (mr_wins / mr_trades * 100) if mr_trades > 0 else 0
        mom_wr = (mom_wins / mom_trades * 100) if mom_trades > 0 else 0
        
        # MODIFIED: Updated portfolio display to show position utilization
        total_position_value = sum(pos.size for pos in self.positions.values())
        position_utilization = (total_position_value / self.capital * 100) if self.capital > 0 else 0
        
        text = (
            f"[bold]Portfolio & Regime[/]\n"
            f"Equity: [bold green]${eq:,.2f}[/] ([{pnl_color}]{ret_pct:+.2f}%[/])\n"
            f"Positions: {len(self.positions)}/{self.config.max_positions} | Utilization: {position_utilization:.1f}%\n"
            f"Regime: {regime_emoji}  [{regime_color}]{self.active_regime}[/] ({regime_mode_str})\n"
            f"GARCH Vol Ratio: {self.regime_confidence:.2f}\n"
            f"History: {' → '.join(list(self.regime_history)[-3:])}\n\n"
            f"[bold]Overall Performance[/]\n"
            f"Net P&L: [bold {pnl_color}]${m.net_pnl:+,.2f}[/] | Trades: {m.total_trades} | Win Rate: {m.win_rate:.1f}%\n\n"
            f"[bold]Regime Performance[/]\n"
            f"📈 [yellow]Momentum[/]: {mom_trades} trades | WR: {mom_wr:.1f}% | P&L: ${m.momentum_pnl:+.2f}\n"
            f"↩️  [cyan]Mean Rev[/]:  {mr_trades} trades | WR: {mr_wr:.1f}% | P&L: ${m.mean_reversion_pnl:+.2f}"
        )
        return Panel(Text.from_markup(text), title="Performance", border_style="yellow", box=box.ROUNDED)

    def _create_positions_table(self) -> Panel:
        table = Table(box=box.ROUNDED, expand=True)
        # MODIFIED: Adjusted column layout for better space utilization with more positions
        cols = ["Symbol", "Side", "Reg", "Size", "Entry", "P&L", "%", "Time"]
        for col in cols: table.add_column(col, justify="center")
        
        # MODIFIED: Show positions sorted by P&L for better visibility of winners/losers
        sorted_positions = sorted(self.positions.values(), key=lambda p: p.unrealized_pnl, reverse=True)
        
        for pos in sorted_positions:
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            side_color = "green" if pos.side == PositionSide.LONG else "red"
            
            regime_map = {TradingRegime.MOMENTUM.value: "📈", TradingRegime.MEAN_REVERSION.value: "↩️"}
            
            # MODIFIED: Shortened display format to fit more positions
            table.add_row(
                pos.symbol[:6],  # Truncate symbol if too long
                f"[{side_color}]{pos.side.value[0]}[/]",  # Just L or S
                regime_map.get(pos.regime, "❓"),
                f"${pos.size/1000:.1f}k",  # Show in thousands
                f"${pos.entry_price:.3f}",  # 3 decimal places
                f"[{pnl_color}]${pos.unrealized_pnl:+.0f}[/]",  # No decimals for P&L
                f"[{pnl_color}]{pos.get_return_pct():+.1f}%[/]",
                f"{pos.get_duration_minutes():.0f}m"
            )
            
        return Panel(table, title=f"Active Positions ({len(self.positions)}/{self.config.max_positions})", border_style="cyan")

    def _create_vol_ratio_chart(self) -> Panel:
        """Creates a text-based chart for the GARCH volatility ratio history."""
        chart_height = 25  # 20 rows
        chart_width = 60
        history = list(self.vol_ratio_history)
        if not history:
            return Panel("[dim]Awaiting GARCH data...[/dim]", title="GARCH Volatility Ratio", border_style="magenta")

        data_points = history[-chart_width:]
        canvas = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]

        # Draw data points - HARD CODED 0.0 to 2.0 range
        for x, val in enumerate(data_points):
            # Hard clip to 0.0-2.0 range
            if val < 0.0:
                val = 0.0
            elif val > 2.5:
                val = 2.5
            
            # Map to chart row (0.0 = bottom row, 2.0 = top row)
            chart_row = int((val / 2.5) * (chart_height - 1))
            chart_row = max(0, min(chart_height - 1, chart_row))
            
            # Invert for display (top row = index 0)
            display_row = chart_height - 1 - chart_row
            canvas[display_row][x] = "[magenta]•[/]"

        # Create labels - HARD CODED from 2.0 down to 0.0
        labeled_chart = []
        for i in range(chart_height):
            # Hard coded: top row = 2.0, bottom row = 0.0
            label_val = 2.5 - (i / (chart_height - 1)) * 2.5
            label = f"{label_val: >4.1f} │"
            row_content = "".join(canvas[i])
            labeled_chart.append(f"[dim]{label}[/]" + row_content)

        return Panel(Text("\n").join(Text.from_markup(row) for row in labeled_chart),
                     title="GARCH Volatility Ratio Trend (0.0-2.5)", border_style="magenta")

    def _create_footer(self) -> Panel:
        footer_text = ("[bold]Manual Control[/]: Press [bold green]M[/]omentum | [bold green]R[/]eversion | "
                       "[bold green]U[/]nknown | [bold green]A[/]uto (GARCH) then Enter")
        return Panel(Text.from_markup(footer_text), box=box.ROUNDED)

    async def _cleanup(self):
        self.logger.info("🧹 Cleaning up resources...")
        if self.websocket and self.websocket.open: await self.websocket.close()
        self.logger.info("✅ Cleanup complete.")

# ═══════════════════════════════════════════════════════════════════════════
#                                       MAIN
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(config: TradingConfig):
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    file_fmt = logging.Formatter("%(asctime)s|%(name)-20s|%(levelname)-8s| %(message)s", "%H:%M:%S")
    
    # Use a file handler for detailed, persistent logs
    fh = logging.FileHandler(f"betablade_client_v4-4_{int(time.time())}.log", mode='w')
    fh.setFormatter(file_fmt)
    fh.setLevel(log_level)
    
    # The Rich handler will print to the TUI. We don't want duplicate logs in the console
    # before the TUI starts, so we only add the file handler initially.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(fh)

    for lib in ["websockets.client", "asyncio", "arch"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

async def main():
    config = TradingConfig()
    loop = asyncio.get_running_loop()
    client = BetaBladeClient(config, loop)
    client.shutdown_event = asyncio.Event()
    
    def signal_handler(sig):
        if not client.shutdown_event.is_set():
            logging.warning(f"Signal {sig.name} received, shutting down...")
            loop.call_soon_threadsafe(client.shutdown_event.set)
            
    # Start the keyboard listener in a separate daemon thread
    input_thread = threading.Thread(target=client._keyboard_input_loop, daemon=True)
    input_thread.start()
    
    for sig in [signal.SIGINT, signal.SIGTERM]:
        try: loop.add_signal_handler(sig, signal_handler, sig)
        except NotImplementedError: pass

    client_task = asyncio.create_task(client.start())
    shutdown_waiter = asyncio.create_task(client.shutdown_event.wait())
    
    done, pending = await asyncio.wait({client_task, shutdown_waiter}, return_when=asyncio.FIRST_COMPLETED)
    
    for task in pending: task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    logging.info("BETABLADE Client shutdown complete.")

if __name__ == "__main__":
    print("💎 Starting BETABLADE Trading Client V4.4 (Modified for 20 positions)...")
    print("Dashboard will start shortly. Once running, you can type commands in the console.")
    print("\n📦 DEPENDENCY: pip install numpy websockets rich arch\n")
    
    try:
        config = TradingConfig()
        setup_logging(config)
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 BETABLADE Client stopped by user.")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        logging.critical(f"Fatal error in main execution: {e}", exc_info=True)