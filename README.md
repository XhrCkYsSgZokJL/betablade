# 💎 BetaBlade HFT Trading Algorithm

**A sophisticated high-frequency trading system with GARCH-based regime detection and multi-timeframe signal fusion**

## 🚀 Overview

BetaBlade is a professional-grade algorithmic trading system designed for cryptocurrency markets, specifically integrated with Hyperliquid. The system consists of two main components:

- **Server (V3.4.0)**: Real-time market data processing with multi-timeframe signal generation
- **Client (V4.4)**: GARCH-based regime detection with automated position management

## ✨ Key Features

### 🎯 Server Features
- **Multi-Timeframe Analysis**: Simultaneous processing of 1m, 5m, and 10m timeframes
- **Advanced Signal Processing**: PCA-based factor decomposition with Kalman filtering
- **Real-Time Data Stream**: WebSocket connection to Hyperliquid for live market data
- **Consensus Signal Generation**: Weighted fusion of signals across timeframes
- **Live Dashboard**: Rich terminal UI with signal visualization and performance metrics

### 🧠 Client Features
- **GARCH Volatility Modeling**: Regime detection using GARCH(1,1) volatility forecasting
- **Adaptive Position Sizing**: Risk-adjusted position sizing with multiple constraints
- **Enhanced Portfolio Management**: Support for up to 20 simultaneous positions
- **Real-Time Monitoring**: Live dashboard with P&L tracking and regime visualization
- **Manual Override**: Optional manual regime control for discretionary trading

## 📋 Requirements

### Dependencies
```bash
pip install numpy websockets rich arch scikit-learn httpx
```

### System Requirements
- Python 3.8+
- Minimum 4GB RAM
- Stable internet connection for real-time data
- Linux/macOS recommended (Windows supported)

## 🛠️ Installation

1. **Clone or download the files**:
   ```bash
   # Download client.py and server.py to your trading directory
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy websockets rich arch scikit-learn httpx
   ```

3. **Verify installation**:
   ```bash
   python -c "import numpy, websockets, rich, arch, sklearn, httpx; print('✅ All dependencies installed')"
   ```

## 🚦 Quick Start

### 1. Start the Server
```bash
python server.py
```

The server will:
- Connect to Hyperliquid WebSocket feed
- Begin processing market data across multiple timeframes
- Start WebSocket server on `ws://localhost:8889`
- Display live dashboard with signal analysis

### 2. Start the Client
```bash
python client.py
```

The client will:
- Connect to the local server
- Initialize GARCH volatility models
- Begin automated trading based on regime detection
- Display live portfolio dashboard

## 📊 Understanding the System

### Signal Generation Process

1. **Market Data Ingestion**: Real-time price feeds from Hyperliquid
2. **Return Calculation**: Log returns with noise filtering and scaling
3. **Factor Decomposition**: PCA analysis to separate systematic vs. idiosyncratic returns
4. **Kalman Filtering**: Adaptive filtering with velocity estimation and Z-score calculation
5. **Multi-Timeframe Fusion**: Consensus signals across 1m, 5m, and 10m timeframes
6. **Regime Detection**: GARCH-based volatility modeling for market regime identification

### Trading Logic

#### Regime-Based Strategy
- **Momentum Regime**: Trade in direction of consensus signals (long on positive signals)
- **Mean Reversion Regime**: Trade against consensus signals (short on positive signals)
- **Unknown Regime**: Hold positions, close existing trades for risk management

#### Position Management
- **Entry Criteria**: Minimum signal strength (0.2), stability score (0.4), and regime confidence
- **Position Sizing**: 5% maximum per position, $5,000 absolute cap, risk-adjusted sizing
- **Exit Criteria**: Stop loss (1%), take profit (2%), timeout (5 minutes), or signal reversal

## ⚙️ Configuration

### Server Configuration (`TradingConfig` in server.py)
```python
@dataclass(frozen=True)
class TradingConfig:
    # Market data settings
    min_daily_volume: float = 100_000  # Minimum volume filter
    universe_update_interval: int = 300  # Universe refresh (seconds)
    
    # Signal processing
    pca_max_components: int = 2  # PCA factors
    kalman_process_variance: float = 5e-2  # Process noise
    kalman_measurement_variance: float = 5e-5  # Measurement noise
    
    # Smoothing parameters
    kalman_velocity_ema_alpha: float = 0.5  # Velocity smoothing
    kalman_z_score_ema_alpha: float = 0.2  # Z-score smoothing
```

### Client Configuration (`TradingConfig` in client.py)
```python
@dataclass(frozen=True)
class TradingConfig:
    # Portfolio settings
    starting_capital: float = 100_000.0
    max_positions: int = 20
    max_position_size_pct: float = 0.05  # 5% per position
    max_position_size_absolute: float = 5000.0  # $5K max per position
    
    # Risk management
    stop_loss_pct: float = 0.01  # 1% stop loss
    take_profit_pct: float = 0.02  # 2% take profit
    position_timeout_hours: float = 0.0833  # 5 minutes
    
    # Signal thresholds
    min_signal_score: float = 0.2
    min_stability_score: float = 0.4
```

## 🎮 Manual Controls

The client supports manual regime override:

- **M**: Force Momentum regime
- **R**: Force Mean Reversion regime  
- **U**: Force Unknown regime (defensive)
- **A**: Return to automatic GARCH control

Type the letter and press Enter to activate.

## 📈 Dashboard Guide

### Server Dashboard
- **Header**: Uptime, processed ticks, connection status, universe size
- **Signal Table**: Top signals with Z-scores across timeframes
- **Signal Graph**: Live visualization of signal trends for top assets

### Client Dashboard
- **Header**: System status and performance summary
- **Portfolio Panel**: Equity, returns, regime status, and performance metrics
- **Positions Table**: Active positions with P&L and timing information
- **GARCH Chart**: Live volatility ratio visualization (0.0-2.5 range)

## 🔧 Troubleshooting

### Common Issues

**Connection Errors**:
```bash
# Check if server is running
netstat -an | grep 8889

# Restart server if needed
python server.py
```

**Missing Dependencies**:
```bash
# Install missing packages
pip install numpy websockets rich arch scikit-learn httpx
```

**Performance Issues**:
- Reduce `max_positions` in client config
- Increase `universe_update_interval` in server config
- Monitor system resources (CPU/Memory)

### Debug Mode
Enable detailed logging by modifying the `log_level` in configs:
```python
log_level: str = "DEBUG"  # Change from "INFO"
```

## ⚠️ Risk Warnings

### Important Disclaimers
- **Paper Trading**: Test thoroughly before live trading
- **Risk Management**: Never risk more than you can afford to lose
- **Market Conditions**: Performance varies significantly across market regimes
- **Latency Sensitivity**: HFT strategies require low-latency execution
- **Regulatory Compliance**: Ensure compliance with local trading regulations

### Risk Controls
- Built-in position limits and stop losses
- Portfolio-level risk management
- Automatic position closure in unknown regimes
- Timeout-based position management

## 📊 Performance Monitoring

### Key Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns (calculated from P&L history)

### Regime Performance
The system tracks performance separately for:
- Momentum regime trades
- Mean reversion regime trades
- Overall strategy performance

## 🔄 Updates and Maintenance

### Log Files
- Server: `trading_engine_V3_garch_ready_[timestamp].log`
- Client: `betablade_client_v4-4_[timestamp].log`

### Regular Maintenance
- Monitor log files for errors
- Review performance metrics regularly
- Update universe filters based on market conditions
- Calibrate GARCH parameters if needed

## 🤝 Support

For technical issues:
1. Check log files for error messages
2. Verify all dependencies are installed
3. Ensure stable internet connection
4. Monitor system resources

## 📄 License

This software is provided as-is for educational and research purposes. Users are responsible for compliance with applicable regulations and risk management.

---

**⚡ Trade responsibly and may your algorithms be ever profitable! ⚡**