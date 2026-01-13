from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid

from xgboost import XGBRegressor
import shap

# ----------------------------
# Utility: time split
# ----------------------------
def time_split(df: pd.DataFrame, date_col="date", train_end=None, valid_end=None):
    """
    Splits by date (no shuffling).
    - train: dates <= train_end
    - valid: train_end < dates <= valid_end
    - test : dates > valid_end
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    if train_end is None:
        # default: 70% train, 15% valid, 15% test by unique dates
        dates = np.array(sorted(df[date_col].unique()))
        n = len(dates)
        train_end = dates[int(n * 0.70)]
        valid_end = dates[int(n * 0.85)]
    else:
        train_end = pd.to_datetime(train_end, utc=True)

        valid_end = pd.to_datetime(valid_end, utc=True) if valid_end is not None else train_end

    train = df[df[date_col] <= train_end]
    valid = df[(df[date_col] > train_end) & (df[date_col] <= valid_end)]
    test  = df[df[date_col] > valid_end]

    return train, valid, test, train_end, valid_end


# ----------------------------
# Utility: portfolio-style benchmark
# ----------------------------
def top_bottom_spread_by_date(
        df_pred: pd.DataFrame, 
        pred_col="pred", 
        y_col="target", 
        date_col="date",
        n_buckets=10, 
        benchmark="top_minus_bottom", ## Can also be top minus mean
        top_N = None, ## If not none, switch to top N average instead of top decile
        groupby_col=None
    ):
    """
    Computes equal-weight top-decile minus bottom-decile spread per day, then averages.
    Returns:
      daily_spread: Series indexed by date
      avg_spread: float
    """
    dfp = df_pred.copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col], utc=True)

    def _spread(g):
        g = g.dropna(subset=[pred_col, y_col])
        if len(g) < n_buckets * 5:  # skip very small cross-sections
            return np.nan
        g = g.sort_values(pred_col)
        # split into quantile buckets by rank (stable even if many ties)
        g["bucket"] = pd.qcut(np.arange(len(g)), q=n_buckets, labels=False)
        
        if benchmark == "top_minus_bottom" and top_N is None:
            bot = g[g["bucket"] == 0][y_col].mean()
            top = g[g["bucket"] == n_buckets - 1][y_col].mean()
        elif benchmark == "top_minus_bottom" and top_N is not None:
            bot = g.head(top_N)[y_col].mean()
            top = g.tail(top_N)[y_col].mean()
        elif benchmark == "top_minus_mean" and top_N is None:
            bot = g[y_col].mean()
            top = g[g["bucket"] == n_buckets - 1][y_col].mean()
        elif benchmark == "top_minus_mean" and top_N is not None:
            bot = g[y_col].mean()
            top = g.tail(top_N)[y_col].mean()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
           
        return top - bot
    
    if groupby_col is None:
        groupby_col = date_col
    daily = dfp.groupby(groupby_col, sort=True).apply(_spread)
    avg = float(np.nanmean(daily.values))
    return daily, avg


def spearman_ic_by_date(df_pred: pd.DataFrame, pred_col="pred", y_col="target", date_col="date", groupby_col=None):
    """
    Spearman information coefficient (rank correlation) computed cross-sectionally per date.
    Returns:
      daily_ic: Series
      mean_ic: float
    """
    dfp = df_pred.copy()
    dfp[date_col] = pd.to_datetime(dfp[date_col], utc=True)

    def _ic(g):
        g = g.dropna(subset=[pred_col, y_col])
        if len(g) < 20:
            return np.nan
        return g[pred_col].corr(g[y_col], method="spearman")
    
    if groupby_col is None:
        groupby_col = date_col
    daily = dfp.groupby(groupby_col, sort=True).apply(_ic)
    mean_ic = float(np.nanmean(daily.values))
    return daily, mean_ic


@dataclass
class SplitFrames:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    valid_end: pd.Timestamp


class XGBPanelTrainer:
    """
    Convenience trainer for panel (ticker, date) regression + finance-style evaluation.

    Expects external helpers to exist:
      - time_split(df, date_col, train_end=None, valid_end=None) -> (train, valid, test, train_end, valid_end)
      - spearman_ic_by_date(df, pred_col, y_col, date_col) -> (daily_ic: pd.Series, mean_ic: float)
      - top_bottom_spread_by_date(df, pred_col, y_col, date_col) -> (daily_spread: pd.Series, avg_spread: float)
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "target_30d",
        ticker_col: str = "ticker",
        date_col: str = "date",
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.ticker_col = ticker_col
        self.date_col = date_col
        self.random_state = random_state

        self.model: Optional[XGBRegressor] = None
        self.metrics_: Optional[Dict[str, Any]] = None

        # For posthoc analysis
        self.daily_ic_: Optional[pd.Series] = None
        self.daily_spread_: Optional[pd.Series] = None

    # ---------- Public API ----------

    def fit(
        self,
        df: pd.DataFrame,
        xgb_params: Optional[Dict[str, Any]] = None,
        train_end=None,
        valid_end=None,
        eval_verbose: bool | int = False,
        plot: bool = True,
        plot_shap: bool = True,
        shap_sample: int = 2000,
        save_dir: Optional[str | Path] = None,
        save_plots: bool = False,
        show_plots: bool = True
    ) -> Tuple[XGBRegressor, Dict[str, Any]]:
        """
        Train + evaluate.

        Args:
          xgb_params: dict of XGBRegressor hyperparams. Passed at fit-time.
          early_stopping_rounds: if set, uses validation early stopping.
          save_dir: directory for plot saving (and optionally model later, if you add it).
          save_plots: if True, saves plots (IC, spread, SHAP) to save_dir.
          show_plots: if True, displays plots.
        """
        df = self._prep_df(df)

        split = self._split(df, train_end=train_end, valid_end=valid_end)
        self.splits_ = split

        X_train, y_train = split.train[self.feature_cols], split.train[self.target_col]
        X_valid, y_valid = split.valid[self.feature_cols], split.valid[self.target_col]
        X_test, y_test = split.test[self.feature_cols], split.test[self.target_col]

        # Default params (can be overridden by xgb_params)
        params = {
            "n_estimators": 1500,
            "learning_rate": 0.01,
            "max_depth": 2,
            "min_child_weight": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
            "objective": "reg:squarederror",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        if xgb_params:
            params.update(xgb_params)

        self.model = XGBRegressor(**params)

        fit_kwargs = {
            "X": X_train,
            "y": y_train,
            "eval_set": [(X_valid, y_valid)],
            "verbose": eval_verbose,
        }

        self.model.fit(**fit_kwargs)

        # Predictions
        pred_train = self.model.predict(X_train)
        pred_valid = self.model.predict(X_valid)
        pred_test = self.model.predict(X_test)

        # Metrics
        metrics = {
            "train": self._reg_metrics(y_train, pred_train),
            "valid": self._reg_metrics(y_valid, pred_valid),
            "test": self._reg_metrics(y_test, pred_test),
            "split": {"train_end": str(split.train_end), "valid_end": str(split.valid_end)},
            "xgb_params": params,
        }
        # Finance-style metrics on test
        valid_eval = split.valid[[self.ticker_col, self.date_col, self.target_col]].copy()
        valid_eval["pred"] = pred_valid
        valid_eval = valid_eval.rename(columns={self.target_col: "target"})

        _, mean_ic_valid = spearman_ic_by_date(
            valid_eval, pred_col="pred", y_col="target", date_col=self.date_col
        )
        _, avg_spread_valid = top_bottom_spread_by_date(
            valid_eval, pred_col="pred", y_col="target", date_col=self.date_col
        )
        _, avg_topn_valid = top_bottom_spread_by_date(
            valid_eval, pred_col="pred", y_col="target", date_col=self.date_col, top_N=20, benchmark="top_minus_mean"
        )

        metrics["valid"]["mean_spearman_ic"] = float(mean_ic_valid)
        metrics["valid"]["avg_top_minus_bottom_decile"] = float(avg_spread_valid)
        metrics["valid"]["avg_top_minus_mean_topn"] = float(avg_topn_valid)

        # Finance-style metrics on test
        test_eval = split.test[[self.ticker_col, self.date_col, self.target_col]].copy()
        test_eval["pred"] = pred_test
        test_eval = test_eval.rename(columns={self.target_col: "target"})

        daily_ic, mean_ic = spearman_ic_by_date(
            test_eval, pred_col="pred", y_col="target", date_col=self.date_col
        )
        daily_spread, avg_spread = top_bottom_spread_by_date(
            test_eval, pred_col="pred", y_col="target", date_col=self.date_col
        )
        daily_topn, avg_topn = top_bottom_spread_by_date(
            test_eval, pred_col="pred", y_col="target", date_col=self.date_col, top_N=20, benchmark="top_minus_mean"
        )

        self.test_eval_ = test_eval
        self.daily_ic_ = daily_ic
        self.daily_spread_ = daily_spread
        self.daily_topn_ = daily_topn

        metrics["test"]["mean_spearman_ic"] = float(mean_ic)
        metrics["test"]["avg_top_minus_bottom_decile"] = float(avg_spread)
        metrics["test"]["avg_top_minus_mean_topn"] = float(avg_topn)

        self.metrics_ = metrics

        # Plots
        save_path = Path(save_dir) if save_dir is not None else None
        if save_plots and save_path is None:
            raise ValueError("save_plots=True requires save_dir to be provided.")

        if plot:
            self._plot_series(
                series=daily_ic,
                title="Daily Spearman IC (test)",
                ylabel="IC",
                filename="daily_ic.png",
                save_dir=save_path if save_plots else None,
                show=show_plots,
            )
            self._plot_series(
                series=daily_spread,
                title="Top-Decile minus Bottom-Decile Spread (test)",
                ylabel="Spread (avg target return)",
                filename="daily_spread.png",
                save_dir=save_path if save_plots else None,
                show=show_plots,
            )

        if plot_shap:
            self._plot_shap(
                X_test=X_test,
                shap_sample=shap_sample,
                save_dir=save_path if save_plots else None,
                show=show_plots,
            )

        return self.model, metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        df = self._prep_df(df)
        return self.model.predict(df[self.feature_cols])

    # ---------- Internals ----------

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.date_col] = pd.to_datetime(out[self.date_col], utc=True)
        out = out.sort_values([self.date_col, self.ticker_col])
        return out

    def _split(self, df: pd.DataFrame, train_end=None, valid_end=None) -> SplitFrames:
        train, valid, test, te, ve = time_split(
            df, date_col=self.date_col, train_end=train_end, valid_end=valid_end
        )
        return SplitFrames(train=train, valid=valid, test=test, train_end=te, valid_end=ve)

    @staticmethod
    def _reg_metrics(y_true, y_pred) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    @staticmethod
    def _plot_series(
        series: pd.Series,
        title: str,
        ylabel: str,
        filename: str,
        save_dir: Optional[Path],
        show: bool,
    ) -> None:
        fig = plt.figure()
        series.plot()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / filename, bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _plot_shap(
        self,
        X_test: pd.DataFrame,
        shap_sample: int,
        save_dir: Optional[Path],
        show: bool,
    ) -> None:
        if self.model is None:
            return

        X_shap = X_test.copy()
        if len(X_shap) > shap_sample:
            X_shap = X_shap.sample(n=shap_sample, random_state=self.random_state)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_shap)

        # Beeswarm
        plt.figure()
        shap.summary_plot(shap_values, X_shap, show=False)
        plt.title("SHAP Summary (test sample)")

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.gcf().savefig(save_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        else:
            plt.close(plt.gcf())

        # Bar
        plt.figure()
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (bar)")

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.gcf().savefig(save_dir / "shap_bar.png", bbox_inches="tight", dpi=150)

        if show:
            plt.show()
        else:
            plt.close(plt.gcf())
