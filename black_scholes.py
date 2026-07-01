import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar


# Classe base simulada apenas para evitar erro de importação na estrutura de pastas local
class PricingModel:
    pass


class BlackScholesModel(PricingModel):
    """Implementação do modelo Black-Scholes clássico e modificado."""

    SQRT_2PI = math.sqrt(2.0 * math.pi)

    @staticmethod
    def _std_norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _std_norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / BlackScholesModel.SQRT_2PI

    def _bs_d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Preço do modelo Black-Scholes clássico (p=1)."""
        if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma):
            return np.nan
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return np.nan

        d1, d2 = self._bs_d1_d2(S, K, T, r, sigma)

        if option_type.upper() == "CALL":
            return S * self._std_norm_cdf(d1) - K * math.exp(-r * T) * self._std_norm_cdf(d2)
        else:
            return K * math.exp(-r * T) * self._std_norm_cdf(-d2) - S * self._std_norm_cdf(-d1)

    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return 0.0
        d1, _ = self._bs_d1_d2(S, K, T, r, sigma)
        return S * math.sqrt(T) * self._std_norm_pdf(d1)

    def price_modificado(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str, p: float = 1.0) -> float:
        """
        Preço modificado com ajuste p baseado no modelo difusão-advecção-reação.
        p=1 => Reduz-se exatamente ao Black-Scholes padrão.
        """
        if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma) or pd.isna(p):
            return np.nan
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or p <= 0:
            return np.nan

        try:
            # Cálculo do fator multiplicativo e^( (p-1) * sigma^2 * T / 2 )
            exp_arg = (p - 1.0) * (sigma**2) * T / 2.0
            if exp_arg > 700:
                return np.inf
            if exp_arg < -700:
                fator = 0.0
            else:
                fator = math.exp(exp_arg)

            denom = sigma * math.sqrt(max(p, 1e-16) * T)
            if denom <= 0:
                return np.nan

            # Estrutura base de d1 e d2 conforme equações (173) e (174) da dissertação
            base = math.log(S / K) - 0.5 * sigma**2 * T + r * T
            d1 = (base + p * sigma**2 * T) / denom
            d2 = base / denom

            # Restringir d1/d2 para evitar overflow interno na CDF
            d1 = max(min(d1, 40.0), -40.0)
            d2 = max(min(d2, 40.0), -40.0)

            # Execução condicional dependendo do tipo da opção
            if option_type.upper() == "CALL":
                # Equação (172) - Preço da Opção de Compra Modificada
                preco = fator * S * self._std_norm_cdf(d1) - math.exp(-r * T) * K * self._std_norm_cdf(d2)
            else:
                # Equação deduzida via paridade - Preço da Opção de Venda Modificada
                preco = math.exp(-r * T) * K * self._std_norm_cdf(-d2) - fator * S * self._std_norm_cdf(-d1)

            if math.isinf(preco) or math.isnan(preco):
                return np.nan
            return preco

        except (ValueError, OverflowError):
            return np.nan

    def resolve_p_por_preco(self, S: float, K: float, T: float, r: float, sigma: float,
                             preco_mkt: float, option_type: str, p_low: float = 1e-6, p_high: float = 10.0) -> tuple[float, str]:
        """
        Calibra o parâmetro p de forma que price_modificado(...) == preco_mkt.
        Utiliza o Método de Brent (combinação de Bisseção e Secante) ou fallback por SSE.
        """
        if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(r) or pd.isna(sigma) or pd.isna(preco_mkt):
            return np.nan, 'invalid'
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or preco_mkt < 0:
            return np.nan, 'invalid'

        # Função objetivo modificada para passar o option_type
        def f(p: float) -> float:
            val = self.price_modificado(S, K, T, r, sigma, option_type, p)
            if pd.isna(val) or math.isinf(val):
                return np.nan
            return val - preco_mkt

        def safe_eval(p: float) -> float:
            try:
                v = f(p)
                if pd.isna(v) or math.isinf(v):
                    return np.nan
                return v
            except (ValueError, OverflowError):
                return np.nan

        f_low, f_high = safe_eval(p_low), safe_eval(p_high)

        # Loop de expansão (bracketing) adaptativo
        expand = 0
        while (not np.isfinite(f_low) or not np.isfinite(f_high) or f_low * f_high > 0) and expand < 10:
            p_low = max(1e-12, p_low / 2)
            p_high = min(p_high * 2, 1e6)
            f_low, f_high = safe_eval(p_low), safe_eval(p_high)
            expand += 1

        # PLANO A: Se houver mudança de sinal no intervalo, usa busca por raízes exata
        if np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high <= 0:
            try:
                p_star = brentq(lambda x: safe_eval(x), p_low, p_high, xtol=1e-10, rtol=1e-10, maxiter=200)
                if np.isfinite(p_star):
                    return p_star, 'root'
            except (ValueError, RuntimeError, OverflowError):
                pass

        if not np.isfinite(f_low) or not np.isfinite(f_high):
            return np.nan, 'invalid'

        # PLANO B: Se não bracketar perfeitamente, minimiza a soma dos erros quadráticos (SSE)
        def sse(p: float) -> float:
            d = safe_eval(p)
            if not np.isfinite(d):
                return float('inf')
            if abs(d) > 1e150:
                return 1e300
            return d * d

        res = minimize_scalar(sse, bounds=(p_low, p_high), method='bounded', options={'xatol': 1e-9})
        return res.x, 'sse'