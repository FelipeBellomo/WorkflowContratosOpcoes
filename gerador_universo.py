from __future__ import annotations

import io
import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import os
import re
import requests
import numpy as np
import yfinance as yf
from urllib.parse import urljoin

from black_scholes import BlackScholesModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class OpcaoRegistro:
    ticker: str
    codigo_mercado: str
    preco_fechamento: Optional[float]
    volume_financeiro: float
    strike: Optional[float]
    ativo_objeto: str
    ativo_mae: Optional[str] = None
    data_pregao: Optional[date] = None
    data_vencimento: Optional[date] = None
    dias_uteis: Optional[int] = None
    T: Optional[float] = None
    sigma: Optional[float] = None


def get_ultimo_dia_util(ref: Optional[date] = None) -> date:
    today = ref or date.today()
    # target is yesterday business day (D-1); if today is Monday, target Friday
    candidate = today - timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Saturday,6=Sunday
        candidate -= timedelta(days=1)
    return candidate


def build_possible_urls(date_obj: date) -> List[str]:
    d = date_obj.strftime("%d%m%Y")
    base_name = f"COTAHIST_D{d}"
    # try common historical paths used by B3/BM&FBOVESPA
    return [
        f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{base_name}.ZIP",
        f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{base_name}.zip",
        f"https://bvmf.bovespa.com.br/InstDados/SerHist/{base_name}.ZIP",
        f"https://bvmf.bovespa.com.br/InstDados/SerHist/{base_name}.zip",
        f"https://www.b3.com.br/pesquisapregao/Download/ArquivosHistoricos/{base_name}.ZIP",
        f"https://www.b3.com.br/pesquisapregao/Download/ArquivosHistoricos/{base_name}.zip",
    ]


def download_cotahist_zip(date_obj: date, timeout: int = 30) -> Optional[bytes]:
    urls = build_possible_urls(date_obj)
    for url in urls:
        try:
            LOGGER.info(f"Tentando baixar COTAHIST de {date_obj.isoformat()} via {url}")
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                LOGGER.info("Arquivo baixado com sucesso")
                return resp.content
            LOGGER.warning("URL disponível mas sem conteúdo: %s (status=%s)", url, resp.status_code)
        except requests.RequestException as exc:
            LOGGER.debug("Falha ao acessar %s: %s", url, exc)
    LOGGER.error("Não foi possível encontrar o arquivo COTAHIST para %s em URLs testadas", date_obj.isoformat())
    return None


def extract_txt_from_zip(zip_bytes: bytes, output_path: Optional[str] = None) -> Optional[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            txt_name = None
            for name in zf.namelist():
                if name.upper().endswith('.TXT'):
                    txt_name = name
                    break
            if not txt_name:
                LOGGER.error("Nenhum arquivo .TXT encontrado dentro do zip")
                return None
            with zf.open(txt_name) as f:
                data = f.read()
                txt = data.decode('latin-1')
                if output_path:
                    with open(output_path, 'w', encoding='latin-1') as fh:
                        fh.write(txt)
                    LOGGER.info("TXT extraído e salvo em %s", output_path)
                return txt
    except zipfile.BadZipFile:
        LOGGER.exception("Arquivo zip inválido")
        return None


def parse_cotahist_txt(txt: str) -> List[OpcaoRegistro]:
    linhas = txt.splitlines()
    registros: List[OpcaoRegistro] = []
    for idx, raw in enumerate(linhas):
        if not raw:
            continue
        if not raw.startswith('01'):
            continue
        line = raw
        try:
            ticker = line[12:24].strip()
            codigo_mercado = line[23:27].strip()
            data_pregao_raw = line[2:10].strip()
            data_vencimento_raw = line[202:210].strip()
            preco_raw = line[108:122].strip()
            volume_raw = line[169:188].strip()
            strike_raw = line[187:201].strip()
            ativo_objeto = line[26:39].strip()

            data_pregao = None
            data_vencimento = None
            dias_uteis = None
            T = None
            if data_pregao_raw.isdigit() and len(data_pregao_raw) == 8:
                data_pregao = date(
                    int(data_pregao_raw[0:4]),
                    int(data_pregao_raw[4:6]),
                    int(data_pregao_raw[6:8]),
                )
            if data_vencimento_raw.isdigit() and len(data_vencimento_raw) == 8:
                data_vencimento = date(
                    int(data_vencimento_raw[0:4]),
                    int(data_vencimento_raw[4:6]),
                    int(data_vencimento_raw[6:8]),
                )
            if data_pregao and data_vencimento:
                dias_uteis = int(np.busday_count(data_pregao, data_vencimento))
                T = dias_uteis / 252.0
                if T <= 0:
                    T = 1e-6
            # Se a data de vencimento no campo estiver claramente inválida, busca um fallback mais confiável
            if data_vencimento and data_vencimento.year >= 9000:
                data_vencimento = None
                T = None
                dias_uteis = None

            preco_fechamento: Optional[float]
            if preco_raw and preco_raw.isdigit():
                preco_fechamento = int(preco_raw) / 1000.0
            else:
                try:
                    preco_fechamento = float(preco_raw) / 1000.0
                except Exception:
                    preco_fechamento = None

            try:
                volume_financeiro = int(volume_raw) / 100.0 if volume_raw and volume_raw.strip() else 0.0
            except ValueError:
                try:
                    volume_financeiro = float(volume_raw.replace(',', '.')) / 100.0
                except Exception:
                    volume_financeiro = 0.0

            try:
                strike = int(strike_raw) / 100.0 if strike_raw and strike_raw.strip() else None
            except ValueError:
                try:
                    strike = float(strike_raw.replace(',', '.')) / 100.0
                except Exception:
                    strike = None

            if codigo_mercado != '010' and (strike is None or strike <= 0 or strike > 1000):
                LOGGER.debug('Linha %s descartada por strike inválido para %s: %s', idx + 1, codigo_mercado, strike)
                continue
            if preco_fechamento is None or preco_fechamento <= 0 or preco_fechamento > 500:
                LOGGER.debug('Linha %s descartada por fechamento inválido: %s', idx + 1, preco_fechamento)
                continue

            registros.append(OpcaoRegistro(
                ticker=ticker,
                codigo_mercado=codigo_mercado,
                preco_fechamento=preco_fechamento,
                volume_financeiro=volume_financeiro,
                strike=strike,
                ativo_objeto=ativo_objeto,
                data_pregao=data_pregao,
                data_vencimento=data_vencimento,
                dias_uteis=dias_uteis,
                T=T,
            ))
        except Exception:
            LOGGER.debug("Falha ao parsear linha %s: %s", idx + 1, raw)
            continue
    return registros


def write_debug_artifacts(txt: str, registros: List[OpcaoRegistro], target: date) -> None:
    try:
        # Save a small preview of the raw TXT for inspection
        preview_path = f"cotahist_preview_{target.strftime('%Y%m%d')}.txt"
        with open(preview_path, 'w', encoding='latin-1') as pf:
            for i, line in enumerate(txt.splitlines()[:500]):
                pf.write(f"{i+1:04d}: {line}\n")
        LOGGER.info("Preview salvo em %s", preview_path)

        # Save simple stats about parsed records
        total = len(registros)
        mercado_counts: Dict[str, int] = {}
        ativos_set = set()
        for r in registros:
            mercado_counts[r.codigo_mercado] = mercado_counts.get(r.codigo_mercado, 0) + 1
            if r.ativo_objeto:
                ativos_set.add(r.ativo_objeto)

        stats = {
            'total_registros_01': total,
            'mercado_counts': mercado_counts,
            'ativos_distintos': sorted(list(ativos_set))[:200],
        }
        stats_path = f"parsed_stats_{target.strftime('%Y%m%d')}.json"
        with open(stats_path, 'w', encoding='utf-8') as sf:
            json.dump(stats, sf, ensure_ascii=False, indent=2)
        LOGGER.info("Estatísticas gravadas em %s", stats_path)

        # Save a small sample of records where codigo_mercado == '070'
        sample = [
            {
                'ticker': r.ticker,
                'codigo_mercado': r.codigo_mercado,
                'preco_fechamento': r.preco_fechamento,
                'volume_financeiro': r.volume_financeiro,
                'strike': r.strike,
                'ativo_objeto': r.ativo_objeto,
            }
            for r in registros if r.codigo_mercado == '070'
        ][:100]
        sample_path = f"sample_070_{target.strftime('%Y%m%d')}.json"
        with open(sample_path, 'w', encoding='utf-8') as sf:
            json.dump(sample, sf, ensure_ascii=False, indent=2)
        LOGGER.info("Amostra de registros '070' gravada em %s (n=%d)", sample_path, len(sample))
    except Exception:
        LOGGER.exception("Falha ao gravar artefatos de debug")


def mapear_volatilidades() -> Dict[str, float]:
    ativos_base = ['PETR4', 'VALE3', 'ITUB4']
    vol_map: Dict[str, float] = {}
    for ativo_mae in ativos_base:
        ticker_yf = f"{ativo_mae}.SA"
        try:
            df = yf.download(ticker_yf, period='3mo', progress=False)
            if df.empty:
                raise ValueError('DataFrame vazio retornado do yfinance')
            if 'Close' not in df:
                raise ValueError('Close não encontrado no DataFrame')
            closes = df['Close'].dropna()
            if closes.empty:
                raise ValueError('Close vazio após dropna')
            retornos = np.log(closes / closes.shift(1)).dropna()
            if retornos.empty:
                raise ValueError('Retornos vazios após shift')
            sigma = retornos.std().item() * np.sqrt(252)
            vol_map[ativo_mae] = float(sigma)
        except Exception as e:
            LOGGER.warning('Não foi possível mapear volatilidade para %s, usando fallback: %s', ativo_mae, str(e))
    defaults = {'PETR4': 0.35, 'VALE3': 0.35, 'ITUB4': 0.35}
    for ativo_mae, default_sigma in defaults.items():
        vol_map.setdefault(ativo_mae, default_sigma)
    LOGGER.info('Volatilidades mapeadas: %s', vol_map)
    return vol_map


def classificar_ativo_mae(ativo_objeto: str) -> Optional[str]:
    norm = ''.join(ch for ch in (ativo_objeto or '').upper() if ch.isalnum())
    if 'PETR' in norm:
        return 'PETR4'
    if 'VALE' in norm:
        return 'VALE3'
    if 'ITUB' in norm:
        return 'ITUB4'
    return None


def mapear_precos_subjacente(registros: List[OpcaoRegistro]) -> Dict[str, float]:
    ativos_base = ['PETR4', 'VALE3', 'ITUB4']
    spot_map: Dict[str, float] = {}
    for r in registros:
        if r.codigo_mercado == '010' and r.ticker in ativos_base and r.preco_fechamento is not None:
            spot_map[r.ticker] = r.preco_fechamento
    LOGGER.info('Preços de subjacente mapeados: %s', spot_map)
    return spot_map


def filtrar_opcoes(registros: List[OpcaoRegistro], ativos_mae: List[str], vol_map: Dict[str, float]) -> List[OpcaoRegistro]:
    resultado: List[OpcaoRegistro] = []
    counts: Dict[str, int] = {name: 0 for name in ativos_mae}
    for r in registros:
        if r.codigo_mercado != '070':
            continue
        ativo_mae = classificar_ativo_mae(r.ativo_objeto)
        if ativo_mae in ativos_mae:
            r.ativo_mae = ativo_mae
            r.sigma = vol_map.get(ativo_mae)
            resultado.append(r)
            counts[ativo_mae] += 1
    LOGGER.info("Registros por ativo mãe encontrados: %s", counts)
    return resultado


def top_n_por_ativo(registros: List[OpcaoRegistro], n: int = 20) -> List[OpcaoRegistro]:
    por_ativo: Dict[str, List[OpcaoRegistro]] = {}
    for r in registros:
        chave = r.ativo_mae or r.ativo_objeto
        por_ativo.setdefault(chave, []).append(r)
    resultado: List[OpcaoRegistro] = []
    for ativo, items in por_ativo.items():
        items_sorted = sorted(items, key=lambda x: x.volume_financeiro, reverse=True)
        resultado.extend(items_sorted[:n])
    return resultado


def detectar_tipo_opcao(ativo_objeto: str) -> str:
    if not ativo_objeto:
        return 'CALL'
    obj = ativo_objeto.upper()
    if '/PT' in obj or 'PUT' in obj:
        return 'PUT'
    return 'CALL'


def calcular_p_otimo(
    ticker: str,
    preco_fechamento: Optional[float],
    strike: Optional[float],
    ativo_objeto: str,
    subjacente_preco: Optional[float],
    T: Optional[float],
    sigma: Optional[float],
) -> float:
    if preco_fechamento is None or strike is None or subjacente_preco is None or T is None or sigma is None:
        # print(f"Dados insuficientes para calcular p_otimo para {ticker}: preco_fechamento={preco_fechamento}, strike={strike}, subjacente_preco={subjacente_preco}, T={T}, sigma={sigma}")  
        return float('nan')
    option_type = detectar_tipo_opcao(ativo_objeto)
    model = BlackScholesModel()

    S = subjacente_preco
    K = strike
    r = 0.1425

    p_star, _ = model.resolve_p_por_preco(S, K, T, r, sigma, preco_fechamento, option_type)
    if p_star is None or not isinstance(p_star, float) or p_star != p_star:
        return float('nan')
    return p_star


def main() -> None:
    target = get_ultimo_dia_util()
    LOGGER.info("Data alvo (D-1): %s", target.isoformat())
    # If user provided a local COTAHIST zip (manual download), prefer it for debugging.
    local_env = os.getenv('LOCAL_COTAHIST_ZIP')
    local_path = local_env or 'COTAHIST_b3_download.zip'
    zip_bytes: Optional[bytes] = None
    if os.path.exists(local_path):
        try:
            LOGGER.info("Usando arquivo local COTAHIST: %s", local_path)
            with open(local_path, 'rb') as f:
                zip_bytes = f.read()
        except Exception:
            LOGGER.exception("Falha ao ler %s", local_path)

    if zip_bytes is None:
        zip_bytes = download_cotahist_zip(target)
    if not zip_bytes:
        LOGGER.error("Arquivo COTAHIST não disponível para %s. Encerrando sem erro.", target.isoformat())
        return

    txt_path = f"COTAHIST_D{target.strftime('%d%m%Y')}.TXT"
    txt = extract_txt_from_zip(zip_bytes, output_path=txt_path)
    if not txt:
        LOGGER.error("Não foi possível extrair TXT do zip. Encerrando.")
        return

    vol_map = mapear_volatilidades()
    registros = parse_cotahist_txt(txt)
    print(f"Total de registros parseados: {len(registros)}")
    spot_map = mapear_precos_subjacente(registros)
    write_debug_artifacts(txt, registros, target)
    ativos_alvo = ['PETR4', 'VALE3', 'ITUB4']
    registros_filtrados = filtrar_opcoes(registros, ativos_alvo, vol_map)

    if not registros_filtrados:
        LOGGER.warning("Nenhuma opção encontrada para ativos alvo em %s", target.isoformat())

    top60 = top_n_por_ativo(registros_filtrados, n=20)

    output: List[Dict] = []

    for r in top60:
        subjacente_preco = spot_map.get(r.ativo_mae)
        p = calcular_p_otimo(
            r.ticker,
            r.preco_fechamento,
            r.strike,
            r.ativo_objeto,
            subjacente_preco,
            r.T,
            r.sigma,
        )
        output.append({
            'ticker': r.ticker,
            'codigo_mercado': r.codigo_mercado,
            'preco_fechamento': r.preco_fechamento,
            'volume_financeiro': r.volume_financeiro,
            'strike': r.strike,
            'ativo_objeto': r.ativo_objeto,
            'ativo_mae': r.ativo_mae,
            'dias_uteis': r.dias_uteis,
            'T': r.T,
            'sigma': r.sigma,
            'p_otimo': p,
        })

    try:
        total_calculado = len(output)
        quantidade_saudavel = sum(1 for item in output if isinstance(item.get('p_otimo'), float) and 0.5 <= item['p_otimo'] <= 1.5)
        porcentagem = (quantidade_saudavel / total_calculado * 100.0) if total_calculado else 0.0
        LOGGER.info('[AUDITORIA] Resumo da Calibração: %d de %d contratos (%.1f%%) estão na faixa normal (0.5-1.5)',
                     quantidade_saudavel, total_calculado, porcentagem)
        with open('top_opcoes_d1.json', 'w', encoding='utf-8') as fh:
            json.dump({'data_alvo': target.isoformat(), 'resultados': output}, fh, ensure_ascii=False, indent=2)
        LOGGER.info("Arquivo top_opcoes_d1.json salvo com %d registros", len(output))
    except Exception:
        LOGGER.exception("Falha ao salvar arquivo de saída")


if __name__ == '__main__':
    main()
