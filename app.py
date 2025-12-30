import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st
import json

# ================== Paths & Kronos imports ==================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent   # ← 关键修改

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402

# ================== Global paths & constants ==================
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "cache" / "energy_multi"

ASSETS = ["WTI", "BRENT", "RBOB", "HO", "NG"]

DEFAULT_LOOKBACK = 256
ADAPTER_HORIZON = 5
PRED_LEN = 5

ADAPTER_CKPT = (
    PROJECT_ROOT
    / "checkpoints"
    / "energy_multi"
    / "kronos_adapter_v0"
    / "kronos_adapter_best_v0.pt"
)

START_DATE_2025 = pd.Timestamp("2025-01-01")
END_DATE_2025 = pd.Timestamp("2025-12-26")

FUTURE_SPECS = {
    "WTI":   {"tick_size": 0.01,   "tick_value": 10.0, "slippage_ticks": 2},
    "BRENT": {"tick_size": 0.01,   "tick_value": 10.0, "slippage_ticks": 2},
    "RBOB":  {"tick_size": 0.0001, "tick_value": 4.2,  "slippage_ticks": 3},
    "HO":    {"tick_size": 0.0001, "tick_value": 4.2,  "slippage_ticks": 3},
    "NG":    {"tick_size": 0.001,  "tick_value": 10.0, "slippage_ticks": 4},
}
FEE_PER_SIDE = 2.5
VOL_WINDOW = 20
K_SIG = 0.4
TREND_WINDOW = 200
WINDOWS_FOR_PLOT = 250


# ================== Basic utilities ==================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_raw_ohlcva(symbol: str) -> pd.DataFrame:
    csv_path = RAW_DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV for {symbol} not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df.set_index("date", inplace=True)

    df = df[["open", "high", "low", "close", "volume"]].copy()
    mean_price = df[["open", "high", "low", "close"]].mean(axis=1)
    mean_price = mean_price.ffill().fillna(0.0)
    df["amount"] = df["volume"].fillna(0.0) * mean_price
    df = df.dropna()
    return df


class KronosAdapter(nn.Module):
    def __init__(self, horizon: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@st.cache_resource
def load_predictor_and_adapter_cached():
    """
    Load tokenizer + Kronos model + local adapter only once.
    """
    device = get_device()
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    kronos_model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    kronos_model.eval()
    predictor = KronosPredictor(kronos_model, tokenizer, device=str(device))

    ckpt = torch.load(ADAPTER_CKPT, map_location=device)
    adapter = KronosAdapter(horizon=ADAPTER_HORIZON, hidden_dim=64).to(device)
    adapter.load_state_dict(ckpt["model_state_dict"])
    adapter.eval()
    return device, predictor, adapter


def section_header(title: str):
    st.markdown(
        f"""
        <hr style="
            border: none;
            height: 1px;
            background-color: #d1d5db;
            margin-top: 1.2rem;
            margin-bottom: 1.2rem;
        ">

        <h2 style="
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: -0.2rem;
        ">{title}</h2>
        """,
        unsafe_allow_html=True,
    )

# ================== Page 1: full-year forecast (from cache) ==================
@st.cache_data
def load_full_year_cache():
    results: Dict[str, Tuple[pd.DataFrame, Dict]] = {}
    for sym in ASSETS:
        df_path = CACHE_DIR / f"full_year_{sym}.parquet"
        m_path = CACHE_DIR / f"full_year_{sym}_metrics.json"
        if not df_path.exists() or not m_path.exists():
            continue
        df_res = pd.read_parquet(df_path)
        with open(m_path, "r") as f:
            metrics = json.load(f)
        results[sym] = (df_res, metrics)
    return results


def page_full_year():
    section_header("How the model tracks the market across 2025")
    st.caption(
        "This view compares the model’s one-day-ahead forecast with the realised daily close "
        "over the full 2025 trading year."
    )

    all_results = load_full_year_cache()
    if not all_results:
        st.error(
            "No cached results were found.\n\n"
            "Please run:\n"
            "`python finetune/energy_multi/kronos_adapter/precompute_full_year_cache.py`"
        )
        return

    symbol = st.selectbox("Instrument", ASSETS, index=0)

    df_res, _ = all_results.get(symbol, (None, None))
    if df_res is None or df_res.empty:
        st.warning("No cached data for this instrument.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_res["date"], df_res["true_close"].values, label="Actual close")
    ax.plot(
        df_res["date"],
        df_res["kronos_close_1d"].values,
        label="Kronos forecast",
        linestyle="--",
    )
    ax.plot(
        df_res["date"],
        df_res["adapter_close_1d"].values,
        label="Adapter forecast",
        linestyle="-",
    )
    ax.set_title(f"{symbol} – 1-day ahead forecast vs actual (2025)")
    ax.set_ylabel("Close price")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig)
    st.caption(
        f"Forecast horizon: 1 trading day ahead. "
        f"Backtest window: {START_DATE_2025.date()} – {END_DATE_2025.date()}."
    )


# ================== Page 2: short-term local window forecast ==================
def _window_days_from_lookback(lookback: int) -> int:
    """Map the lookback choice to a display window (calendar days)."""
    if lookback <= 21:   # ~1 month
        return 30
    if lookback <= 63:   # ~3 months
        return 90
    if lookback <= 126:  # ~6 months
        return 180
    return 365           # ~12 months


def compute_local_window(
    symbol: str,
    mode_key: str,  # "teacher" | "autoregressive"
    lookback: int,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    device: torch.device,
    predictor: KronosPredictor,
    adapter: KronosAdapter,
):
    """
    mode_key:
      - "teacher": Anchored to actual prices (one-step)
      - "autoregressive": Let the model run forward (directional)
    """
    np.random.seed(42)
    torch.manual_seed(42)

    df = load_raw_ohlcva(symbol)

    window_days = _window_days_from_lookback(lookback)
    window_start = pred_start - pd.Timedelta(days=window_days)
    window_end = pred_end + pd.Timedelta(days=5)

    window_df = df.loc[window_start:window_end]
    if window_df.empty:
        return None, None

    window_dates = window_df.index
    window_closes = window_df["close"].values.astype(float)

    pred_dates = df.loc[pred_start:pred_end].index
    if len(pred_dates) == 0:
        return None, None

    # ---------- 1. Kronos & Adapter forecasts ----------
    if mode_key == "teacher":
        # Anchored: always use true history
        kronos_preds = []
        adapter_preds = []

        for date_t in pred_dates:
            idx = df.index.get_loc(date_t)
            end_ctx = idx - 1
            start_ctx = end_ctx - lookback + 1
            if start_ctx < 0:
                continue

            df_ctx = df.iloc[start_ctx: end_ctx + 1]
            x = df_ctx[["open", "high", "low", "close", "volume", "amount"]].copy()
            base_close = float(x["close"].iloc[-1])
            log_base = np.log(base_close)

            y_ts = pd.date_range(start=date_t, periods=ADAPTER_HORIZON, freq="1D")

            with torch.no_grad():
                pred_df_k = predictor.predict(
                    df=x,
                    x_timestamp=x.index.to_series(),
                    y_timestamp=pd.Series(y_ts),
                    pred_len=ADAPTER_HORIZON,
                    T=1.0,
                    top_p=1.0,  # deterministic
                    sample_count=1,
                    verbose=False,
                )
            close_pred_k = pred_df_k["close"].values.astype(float)
            kronos_logret = np.log(close_pred_k) - log_base
            kronos_close_t = base_close * np.exp(kronos_logret[0])
            kronos_preds.append(kronos_close_t)

            kronos_logret_tensor = (
                torch.from_numpy(kronos_logret[None, :]).float().to(device)
            )
            with torch.no_grad():
                adapter_logret_tensor = adapter(kronos_logret_tensor)
            adapter_logret = adapter_logret_tensor.cpu().numpy()[0]
            adapter_close_t = base_close * np.exp(adapter_logret[0])
            adapter_preds.append(adapter_close_t)

        kronos_preds = np.array(kronos_preds, dtype=float)
        adapter_preds = np.array(adapter_preds, dtype=float)

    else:
        # Free-running: feed previous forecasts back into the context
        working_kronos = df.copy()
        working_adapter = df.copy()
        kronos_preds = []
        adapter_preds = []

        for date_t in pred_dates:
            # Kronos AR
            idx_k = working_kronos.index.get_loc(date_t)
            end_ctx_k = idx_k - 1
            start_ctx_k = end_ctx_k - lookback + 1
            if start_ctx_k < 0:
                continue
            df_ctx_k = working_kronos.iloc[start_ctx_k: end_ctx_k + 1]
            x_k = df_ctx_k[
                ["open", "high", "low", "close", "volume", "amount"]
            ].copy()
            base_close_k = float(x_k["close"].iloc[-1])
            log_base_k = np.log(base_close_k)

            y_ts = pd.date_range(start=date_t, periods=ADAPTER_HORIZON, freq="1D")
            with torch.no_grad():
                pred_df_k = predictor.predict(
                    df=x_k,
                    x_timestamp=x_k.index.to_series(),
                    y_timestamp=pd.Series(y_ts),
                    pred_len=ADAPTER_HORIZON,
                    T=1.0,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
            close_pred_k = pred_df_k["close"].values.astype(float)
            kronos_logret = np.log(close_pred_k) - log_base_k
            kronos_close_t = base_close_k * np.exp(kronos_logret[0])
            kronos_preds.append(kronos_close_t)
            working_kronos.at[date_t, "close"] = kronos_close_t

            # Adapter AR
            idx_a = working_adapter.index.get_loc(date_t)
            end_ctx_a = idx_a - 1
            start_ctx_a = end_ctx_a - lookback + 1
            if start_ctx_a < 0:
                continue
            df_ctx_a = working_adapter.iloc[start_ctx_a: end_ctx_a + 1]
            x_a = df_ctx_a[
                ["open", "high", "low", "close", "volume", "amount"]
            ].copy()
            base_close_a = float(x_a["close"].iloc[-1])
            log_base_a = np.log(base_close_a)

            with torch.no_grad():
                pred_df_a = predictor.predict(
                    df=x_a,
                    x_timestamp=x_a.index.to_series(),
                    y_timestamp=pd.Series(y_ts),
                    pred_len=ADAPTER_HORIZON,
                    T=1.0,
                    top_p=1.0,
                    sample_count=1,
                    verbose=False,
                )
            close_pred_a = pred_df_a["close"].values.astype(float)
            kronos_logret_a = np.log(close_pred_a) - log_base_a

            kronos_logret_tensor = (
                torch.from_numpy(kronos_logret_a[None, :]).float().to(device)
            )
            with torch.no_grad():
                adapter_logret_tensor = adapter(kronos_logret_tensor)
            adapter_logret = adapter_logret_tensor.cpu().numpy()[0]
            adapter_close_t = base_close_a * np.exp(adapter_logret[0])
            adapter_preds.append(adapter_close_t)
            working_adapter.at[date_t, "close"] = adapter_close_t

        kronos_preds = np.array(kronos_preds, dtype=float)
        adapter_preds = np.array(adapter_preds, dtype=float)

    # ---------- 2. Plot with coloured dashed lines & arrows ----------
    true_pred = df["close"].loc[pred_dates].values.astype(float)

    color_truth = "tab:blue"
    color_kronos = "tab:green"
    color_adapter = "tab:red"

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        window_dates,
        window_closes,
        label="Actual close",
        color=color_truth,
        linewidth=1.5,
    )
    ax.plot(
        pred_dates,
        kronos_preds,
        linestyle="--",
        label="Kronos forecast",
        color=color_kronos,
        linewidth=1.5,
    )
    ax.plot(
        pred_dates,
        adapter_preds,
        linestyle="--",
        label="Adapter forecast",
        color=color_adapter,
        linewidth=1.5,
    )

    # Arrows only in free-running mode
    if mode_key == "autoregressive" and len(pred_dates) > 0:
        idx_first_pred = df.index.get_loc(pred_dates[0])
        if isinstance(idx_first_pred, (int, np.integer)) and idx_first_pred > 0:
            prev_date = df.index[idx_first_pred - 1]
            prev_price = float(df["close"].iloc[idx_first_pred - 1])
        else:
            prev_date = pred_dates[0]
            prev_price = float(true_pred[0])

        if len(pred_dates) > 1:
            # Kronos arrow (green)
            ax.annotate(
                "",
                xy=(pred_dates[-1], kronos_preds[-1]),
                xytext=(prev_date, prev_price),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2.8,
                    alpha=0.9,
                    color=color_kronos,
                ),
            )
            # Adapter arrow (red)
            ax.annotate(
                "",
                xy=(pred_dates[-1], adapter_preds[-1]),
                xytext=(prev_date, prev_price),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2.8,
                    alpha=0.9,
                    color=color_adapter,
                ),
            )
            # Actual price arrow (blue)
            ax.annotate(
                "",
                xy=(pred_dates[-1], true_pred[-1]),
                xytext=(prev_date, prev_price),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=2.8,
                    alpha=0.9,
                    color=color_truth,
                ),
            )

    ax.axvline(pred_start, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Close price")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    # 返回 None 作为 “指标”，因为 UI 不再展示数值
    metrics = None
    return metrics, fig


def page_local_window():
    section_header("Short-term directional forecasting")
    st.caption(
        "Look at a short forecast window around a date you care about. "
        "You can keep the model anchored to the latest market prices, "
        "or let it run forward to see the directional path it would take."
    )

    device, predictor, adapter = load_predictor_and_adapter_cached()

    col_top = st.columns(3)
    with col_top[0]:
        symbol = st.selectbox("Instrument", ASSETS, index=0)
    with col_top[1]:
        mode_label = st.selectbox(
            "Forecast mode",
            [
                "Teacher forcing (one-step)",
                "Free running (directional)",
            ],
            index=0,
        )
        mode_key = "teacher" if mode_label.startswith("Anchored") else "autoregressive"
    with col_top[2]:
        lookback_label = st.selectbox(
            "Lookback window",
            [
                "1 month",
                "3 months",
                "6 months",
                "12 months",
            ],
            index=1,
        )
        lookback_map = {
            "1 month": 21,
            "3 months": 63,
            "6 months": 126,
            "12 months": 252,
        }
        lookback = lookback_map[lookback_label]

    col_date = st.columns(2)
    with col_date[0]:
        pred_start_date = st.date_input(
            "Forecast start date", value=pd.to_datetime("2025-12-05").date()
        )
    with col_date[1]:
        pred_end_date = st.date_input(
            "Forecast end date", value=pd.to_datetime("2025-12-19").date()
        )

    pred_start = pd.Timestamp(pred_start_date)
    pred_end = pd.Timestamp(pred_end_date)
    if pred_end < pred_start:
        st.error("Forecast end date must be on or after the start date.")
        return

    if st.button("Run short-term forecast", type="primary"):
        with st.spinner("Running forecast window..."):
            _, fig = compute_local_window(
                symbol,
                mode_key,
                lookback,
                pred_start,
                pred_end,
                device,
                predictor,
                adapter,
            )

        if fig is None:
            st.warning("Not enough data or history length for this configuration.")
            return

        st.pyplot(fig)
        st.caption(
            f"Mode: {mode_label}. "
            f"Forecast window: {pred_start.date()} – {pred_end.date()}. "
            f"Lookback: {lookback_label} of history."
        )


# ================== Page 3: Strategy backtest (from cache) ==================
@st.cache_data
def load_backtest_cache():
    results = {}
    for sym in ASSETS:
        df_path = CACHE_DIR / f"backtest_{sym}.parquet"
        m_path = CACHE_DIR / f"backtest_{sym}_metrics.json"
        if not df_path.exists() or not m_path.exists():
            continue
        df_res = pd.read_parquet(df_path)
        with open(m_path, "r") as f:
            metrics = json.load(f)
        results[sym] = (df_res, metrics)
    return results


def make_backtest_fig(df_res: pd.DataFrame, symbol: str):
    if df_res is None or df_res.empty:
        return None

    dates = df_res["date"].values
    true_lr = df_res["true_lr"].values
    pnl_b = df_res["pnl_base"].values
    pnl_a = df_res["pnl_adpt"].values

    cum_buyhold = np.exp(np.cumsum(true_lr)) - 1
    cum_base = np.exp(np.cumsum(pnl_b)) - 1
    cum_adpt = np.exp(np.cumsum(pnl_a)) - 1

    exc_b = cum_base - cum_buyhold
    exc_a = cum_adpt - cum_buyhold

    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(dates, cum_buyhold, "--", label="Buy & hold")
    ax1.plot(dates, cum_base, label="Kronos strategy")
    ax1.plot(dates, cum_adpt, label="Adapter strategy")
    ax1.set_title(
        f"{symbol} – model-driven long/short vs buy & hold"
    )
    ax1.set_ylabel("Cumulative return")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(dates, exc_b, label="Kronos – excess over buy & hold")
    ax2.plot(dates, exc_a, label="Adapter – excess over buy & hold")
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_ylabel("Excess return")
    ax2.set_xlabel("Date")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    return fig


def page_strategy_backtest():
    section_header("Backtesting results")
    st.caption(
        "We backtest a simple model-driven long/short strategy. "
        "Each day the model produces a one-day-ahead view on price direction. "
        "Signals are scaled by recent volatility and filtered by a long-term trend "
        "so the strategy leans long in up-trends and short in down-trends, "
        "trading only when the signal is strong enough to overcome costs."
    )

    all_results = load_backtest_cache()
    if not all_results:
        st.error(
            "No backtest cache found.\n\n"
            "Please run:\n"
            "`python finetune/energy_multi/kronos_adapter/precompute_backtest_cache.py`"
        )
        return

    symbol = st.selectbox("Instrument", ASSETS, index=0)

    df_res, metrics = all_results.get(symbol, (None, None))
    if df_res is None or df_res.empty:
        st.warning("No backtest data for this instrument.")
        return

    base_m = metrics["base"]
    adpt_m = metrics["adapter"]

    # 先画图
    fig = make_backtest_fig(df_res, symbol)
    if fig is not None:
        st.pyplot(fig)

    # 再在图下面用小字展示关键指标
    st.caption(
        f"Backtest window: forecast dates restricted to 2025-01-01 – 2025-12-26. "
        f"Returns are based on log-returns with transaction costs embedded."
    )

    # Small, text-only summary of stats
    st.markdown(
        "<small>"
        f"**Kronos strategy** – trades: {base_m['n_trades']}, "
        f"hit rate: {base_m['winrate']:.3f}, "
        f"annualised Sharpe: {base_m['sharpe']:.3f}, "
        f"max drawdown: {base_m['max_dd']:.3f}.<br>"
        f"**Adapter strategy** – trades: {adpt_m['n_trades']}, "
        f"hit rate: {adpt_m['winrate']:.3f}, "
        f"annualised Sharpe: {adpt_m['sharpe']:.3f}, "
        f"max drawdown: {adpt_m['max_dd']:.3f}."
        "</small>",
        unsafe_allow_html=True,
    )


# ================== Streamlit main app ==================
def main():
    st.set_page_config(
        page_title="Transformer-based Energy Futures Price Forecasting",
        layout="wide",
    )

    # --- Tighten top padding & title spacing ---
    st.markdown(
        """
        <style>
            /* Less top padding for the main block */
            .block-container {
                padding-top: 1.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Custom H1 with smaller bottom margin
    st.markdown(
        """
        <h1 style="font-size:2.6rem; font-weight:700; margin-bottom:0.15rem;">
            Transformer-based Energy Futures Price Forecasting
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # Intro text with very small top margin
    st.markdown(
        """
        <p style="font-size:0.98rem; color:#4b5563; margin-top:0.25rem;">
        This dashboard showcases a Transformer-based time-series model (<b>Kronos, https://github.com/shiyu-coder/Kronos</b>) and a fine-tuned
        adapter trained specifically on major energy futures (WTI, Brent, RBOB, Heating Oil, Natural Gas).
        You can explore how the model tracks the market day-to-day, how it behaves over short forecast
        windows, and how its signals perform inside a simple long/short trading rule.
        </p>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Full-year forecast",
            "2. Short-term forecast",
            "3. Strategy backtesting",
        ],
    )

    if page.startswith("1."):
        page_full_year()
    elif page.startswith("2."):
        page_local_window()
    elif page.startswith("3."):
        page_strategy_backtest()



if __name__ == "__main__":
    main()
