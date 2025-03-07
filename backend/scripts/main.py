from constants import *
from utils import *
from flask import Flask, request, jsonify
from flask_cors import CORS

dp = data_proc()
st = Stats()
md = Models()



def simulator(parameters):
    try:
        input = parameters.get("input", "xl")
        modelo = parameters.get("modelo", "garch")
        horizonte = int(parameters.get("horizonte", 24) or 24)
        n_simul = int(parameters.get("n_simul", 5000) or 5000)
        frequencia = parameters.get("frequencia", "M")
        modelo_vol = parameters.get("modelo_vol", "garch")
        modelo_correl = parameters.get("modelo_correl", "ewma")
        dist = parameters.get("dist", "Students")

        print(f"Parâmetros recebidos: {parameters}")

        prices = dp.get_prices(input)
        gl_dict = st.fit_tstudent(prices)
        scal = st.scaling(prices, frequencia)
        kappa = st.kappa(prices, scal)
        volatility = st.volatility(modelo_vol, prices, scal, horizonte, L=None)
        correlation_matrix = st.correlation(modelo_correl, prices, 0.94)
        cholesky_decomposition = st.cholesky(modelo_vol, correlation_matrix, volatility, horizonte)
        long_term_average = st.long_term_average(prices, horizonte, frequencia)
        montecarlo_dict = md.ornstein_uhlenbeck(prices, kappa, long_term_average, horizonte, frequencia, cholesky_decomposition, volatility, n_simul)
        
        with pd.ExcelWriter("output.xlsx") as writer:
            for coluna in montecarlo_dict:
                montecarlo_dict[coluna].to_excel(writer, sheet_name=coluna)

        return "ok"
    except Exception as e:
        print(f"Erro durante a execução da simulação: {e}")
        raise e
