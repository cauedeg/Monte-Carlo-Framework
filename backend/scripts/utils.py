from constants import *

class data_proc:
    def get_prices(self, input):

        data = pd.read_excel("C:/Users/cauedeg/OneDrive/work/git/catalog/famework_mc/backend/inputs/dummy_prices.xlsx")
        data = pd.pivot_table(data, columns = 'code', index = 'date', values = 'value')

        return data

    def calculate_end_date(self, start_date, periods, frequencia):  

        'Calculo do range de datas através da frequencia escolhida'
        if frequencia == 'D':
            offset = DateOffset(days=periods)
        elif frequencia == 'W':
            offset = DateOffset(weeks=periods)
        elif frequencia == 'ME':
            offset = DateOffset(months=periods)
        elif frequencia == 'Q':
            offset = DateOffset(quarters=periods)
        elif frequencia == 'A':
            offset = DateOffset(years=periods)
        else:
            raise ValueError(f"Invalid frequency: {frequencia}") 

        end_date = start_date + offset 
        return end_date
    
def criar_graficos(montecarlo, nomes):

    dic_dfs = {}
    dic_nomes = list(montecarlo.keys())
    for df in range(len(montecarlo.keys())):
        df = montecarlo[dic_nomes[df]]
        lista_perc_5 = []
        lista_perc_95 = []
        for row in range(len(dic_nomes)):
            row_number = df.iloc[row].sort_values(ascending=True)
            row_number = row_number.reset_index(drop=True)
            row_number_perc_5 = row_number[len(row_number)*0.05]
            row_number_perc_95 = row_number[len(row_number)*0.95]
            lista_perc_5.append(row_number_perc_5)
            lista_perc_95.append(row_number_perc_95)
            row_avg = list(df.mean(axis=1))

        df_final = pd.DataFrame({'5%': lista_perc_5, 'mean': row_avg, 
                                 '95%': lista_perc_95}, index=montecarlo[dic_nomes[df]].index)
    
        dic_dfs.update(df_final)
        df_chart_1 = df_final.reset_index()

        fig, ax = plt.subplots()
        x = df_chart_1['index']
        ax.plot(x, df_chart_1['mean'])
        ax.fill_between(
            x, df_chart_1['5%'], df_chart_1['95%'], color='b', alpha=.15)
        ax.set_ylim(ymin=0)
        ax.set_title(f'{dic_nomes[df]}')
        fig.autofmt_xdate(rotation=45)
        fig.savefig()
        print('pause')

class Stats:

    def correlation(self, modelo, precos, alpha):
        'Retorna matriz de correlação (2D -> Qtde_Ativos x Qtde_Ativos)'
        retornos = (np.log(precos.pct_change(1) + 1)).dropna()
        #Modelo de correl por ewma ou padrão
        if modelo == "ewma":        
            precos = retornos.copy()
            ativos = precos.columns
            precos['N'] = np.arange(len(precos))[::-1]
            precos['lambda'] = (1-alpha)*alpha**(precos['N'])/(1-alpha**len(precos))
            matriz=pd.DataFrame(index = ativos, columns = ativos)
            for i in ativos:
                for j in ativos:
                    precos['pt1'] = precos['lambda']*(precos[i]-precos[i].mean())*(precos[j]-precos[j].mean())
                    pt1 = precos['pt1'].sum()
                    precos['pt2'] =(precos['lambda']*(precos[i]-precos[i].mean())**2)
                    precos['pt3'] =(precos['lambda']*(precos[j]-precos[j].mean())**2)
                    pt4 = np.sqrt(precos['pt2'].sum()*precos['pt3'].sum())
                    correl = pt1/pt4
                    matriz.loc[i,j] = correl
        else:
            matriz=retornos.corr()

        return  matriz

    def volatility(self, modelo, precos,escalonamento,horizonte,L = None):

        'Retorna dicionário com os ativos e respectivas volatilidades (2D -> Qtde_Ativos x Projeção)'
        """
        distribuições : 'StudentsT'
        Calcula a volatilidade das séries de preço de acordo com o modelo selecionado.
        Args:
            modelo = 'garch' - para GARCH(1,1);
            modelo = 'ewma' - para EWMA.
            precos - DataFrame contendo a séries de preços
            frequência - frequência para cálculo da volatilidade (B = dias úteis; D = dias corridos; M = mensal)
            L (opcional) - caso modelo == EWMA, definir o lambda a ser utilizado
            horizonte - definir o horizonte de tempo de projeção para estimação da volatilidade
        Retorna:
            dicionário contendo as volatilidades calculadas para cada período de tempo
        """

        vol = {}
        precos.dropna(inplace = True)
        for col in precos.columns:           
            retornos = np.log(precos[col].pct_change(1) + 1)
            garch_vol = {}

            if modelo == 'garch':
                model = arch_model(retornos.dropna(), mean = 'Zero', vol = 'GARCH', p=1, q=1, dist = 'StudentsT', rescale=False)
                model_fit = model.fit(disp='off')
                yhat = model_fit.forecast(horizon = horizonte, reindex=False)
                garch_vol = yhat.variance.dropna().values[0]**(1/2)
                garch_vol_escalonada = self.scal_vol_garch(garch_vol, horizonte)
                vol[col] = garch_vol_escalonada.copy()
                

            ewma_calc = pd.DataFrame()
            ewma_vol = {}
            if modelo == 'ewma':
                n = len(precos[col])
                ewma_calc['var'] = np.log(precos[col].pct_change(1)+1)**2 
                ewma_calc = ewma_calc.sort_index(ascending = False)
                ewma_calc['wts'] = [(L**(i-1) * (1 - L))  for i in range(1,n+1)]
                ewma_calc['ewma'] = ewma_calc['wts'] * ewma_calc['var']
                ewma_vol = np.sqrt(ewma_calc['ewma'].sum())*np.sqrt(escalonamento)                
                vol[col] = [ewma_vol.copy() for i in range(horizonte)]            
        
        return vol
    
    def vol_ewma(self, prices, scaling, horizonte, L = None):

        vol = {}
        prices.dropna(inplace = True)
        for col in prices.columns:
            ewma_calc = pd.DataFrame()
            ewma_vol = {}
            n = len(prices[col])
            ewma_calc['var'] = np.log(prices[col].pct_change(1)+1)**2 
            ewma_calc = ewma_calc.sort_index(ascending = False)
            ewma_calc['wts'] = [(L**(i-1) * (1 - L))  for i in range(1,n+1)]
            ewma_calc['ewma'] = ewma_calc['wts'] * ewma_calc['var']
            ewma_vol = np.sqrt(ewma_calc['ewma'].sum())*np.sqrt(scaling)                
            vol[col] = [ewma_vol.copy() for i in range(horizonte)]            
        
        return vol
    
    def vol_garch(self, prices, scaling, horizon,L = None):

        vol = {}
        prices.dropna(inplace = True)
        for col in prices.columns:           
            retornos = np.log(prices[col].pct_change(1) + 1)
            garch_vol = {}

            model = arch_model(retornos.dropna(), mean = 'Zero', vol = 'GARCH', p=1, q=1, dist = 'StudentsT', rescale=False)
            model_fit = model.fit(disp='off')
            yhat = model_fit.forecast(horizon = horizonte, reindex=False)
            garch_vol = yhat.variance.dropna().values[0]**(1/2)
            garch_vol_escalonada = self.scal_vol_garch(garch_vol, horizonte)
            vol[col] = garch_vol_escalonada.copy()

        return vol

    def long_term_average(self, precos, horizonte, frequencia, proj=None): 
        """Definição do componente de 'média de longo prazo' do processo de Ornstein-Uhlenbeck 
        (média calculada ou média inputada)"""
        if frequencia == 'M':  # Substituir 'M' por 'ME'
            frequencia = 'ME'
        
        if cen == "s":
            if proj is not None:
                if len(proj) == horizonte:
                    return proj
                else:
                    print("Horizonte de projeção diferente de horizonte definido")
                    return None
            # Calcula a média das colunas    
            data_ini_range = pd.to_datetime(precos.index.max())
            medias = precos.mean(axis=0).to_dict()
            datas = pd.date_range(
                data_ini_range,
                data_proc().calculate_end_date(data_ini_range, horizonte, frequencia),
                freq=frequencia,
                inclusive="right"
            )
            resultado = pd.DataFrame(index=datas, columns=precos.columns)
            for i in resultado.columns:
                resultado[i] = medias[i]

        else:
            data_ini_range = pd.to_datetime(precos.index.max())
            medias = precos.mean(axis=0).to_dict()
            datas = pd.date_range(
                data_ini_range,
                data_proc().calculate_end_date(data_ini_range, horizonte, frequencia),
                freq=frequencia,
                inclusive="right"
            )
            resultado = pd.DataFrame(index=datas, columns=precos.columns)
            for i in resultado.columns:
                resultado[i] = medias[i]

        return resultado

    def cholesky(self, modelo, matriz_correl, volatilidade, horizonte):
        """Retorna matriz de decomposição de Cholesky (2D -> Qtde_Ativos x Qtde_Ativos)"""
        if modelo == 'garch':
            # Inicializar covar como matriz numérica diretamente
            covar = pd.DataFrame(
                np.zeros_like(matriz_correl, dtype=float),
                columns=matriz_correl.columns,
                index=matriz_correl.index,
            )

            chol_list = []
            for p in range(horizonte):
                for i in covar.columns:
                    for j in covar.index:
                        covar.loc[j, i] = matriz_correl.loc[j, i] * volatilidade[i][p] * volatilidade[j][p]

                # Garantir que covar seja numérico (já está inicializado corretamente)
                chol = np.linalg.cholesky(covar.values)
                chol_list.append(chol)

            chol = np.array(chol_list)
            return chol

        if modelo == 'ewma':
            # Inicializar covar como matriz numérica diretamente
            covar = pd.DataFrame(
                np.zeros_like(matriz_correl, dtype=float),
                columns=matriz_correl.columns,
                index=matriz_correl.index,
            )

            for i in covar.columns:
                for j in covar.index:
                    covar.loc[j, i] = matriz_correl.loc[j, i] * volatilidade[i][0] * volatilidade[j][0]

            # Garantir que covar seja numérico (já está inicializado corretamente)
            chol = np.linalg.cholesky(covar.values)
            return chol

    def kappa(self, precos, escalonamento):

        """Calcula o kappa usando regressão linear com statsmodels."""

        kappa = {}
        for col in precos.columns:
            series = precos[col].dropna()
            x = series.shift(1).iloc[1:]  # Valores defasados
            y = series.iloc[1:]           # Valores atuais
            
            # Adicionar uma constante para o intercepto
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            
            alpha = model.params.iloc[1]  # Coeficiente angular
            if alpha > 0:
                k = -np.log(alpha)
            else:
                k = np.nan
            kappa[col] = k * escalonamento

        return kappa
    
    def correl_returns(self, precos, cholesky_matriz):
        'Função para gerar retornos aleatorios e correlacionados'
        ret_corr = []
        for i in cholesky_matriz:
            ret_n_correl = [np.random.standard_normal() for coluna in precos.columns] 
            ret = np.matmul(np.asarray(ret_n_correl).T, i.T)
            ret_corr.append(ret)
        return ret_corr
    
    def correl_returns2(self, precos, cholesky_matriz, gl=None):
        'Função para gerar retornos aleatorios e correlacionados'
        ret_corr = []
        if gl is not None:        
            for i in cholesky_matriz:
                ret_n_correl = [np.random.standard_t(gl[coluna]) for coluna in precos.columns]  
                ret = np.matmul(np.asarray(ret_n_correl).T, i.T)
                ret_corr.append(ret)
            return ret_corr
        else:
            for i in cholesky_matriz:
                ret_n_correl = [np.random.standard_normal() for coluna in precos.columns] 
                ret = np.matmul(np.asarray(ret_n_correl).T, i.T)
                ret_corr.append(ret)
            return ret_corr
    
    def fit_tstudent(self, precos):
        gl_dict ={}
        for coluna in precos.columns:
            gl_dict[coluna] = t.fit(np.log(precos[coluna].pct_change(1) + 1).dropna())[0]
        return gl_dict

    def scaling(self, precos,frequencia):
        'Realiza transformação para a frequencia escolhida independentemente do formato dos dados (ex: semana->meses)'

        freq_dados = pd.infer_freq(precos.index)[0]
        dict_dias = {'D':1,'W':7,'M':30.5,'A':365}
        if freq_dados == frequencia:
            d = 1
            return d
        else:
            d = dict_dias[frequencia]//dict_dias[freq_dados]
            return d
    
    def scal_vol_garch(self, lista_vol, horizonte):    
        'Realiza transformação da vol obtida por garch no periodo dos dados para o periodo de frequência escolhido.'
        part_size = len(lista_vol) // horizonte
        parts = [lista_vol[i:i+part_size] for i in range(0, len(lista_vol), part_size)]
        norms = [np.sqrt(np.sum(np.square(part))) for part in parts]
        return norms
    
class Models:

    
    def ornstein_uhlenbeck(self, precos, kappa, media_lp, horizonte, frequencia, cholesky_matriz, volatilidade, n_simul, gl_dict=None):

        if frequencia == 'M':
            frequencia = 'ME'

        data_ini_range = pd.to_datetime(precos.index.max())
        datas = pd.date_range(
            start=data_ini_range,
            end=data_proc().calculate_end_date(data_ini_range, horizonte, frequencia),
            freq=frequencia,
            inclusive='right'
        )

        montecarlo = {
            serie: pd.DataFrame(index=datas, columns=range(1, n_simul + 1)) for serie in precos.columns
        }

        preco_simul = {serie: [[] for _ in range(n_simul + 1)] for serie in precos.columns}
        
        for coluna in precos.columns:
            ultimo_preco = precos[coluna].iloc[-1]
            for item in range(1, n_simul + 1):
                preco_simul[coluna][item].append(ultimo_preco)

        for j in range(1, n_simul + 1):
            ret_corr = Stats().correl_returns(precos, cholesky_matriz)
            ret_corr = pd.DataFrame(data=ret_corr, columns=precos.columns)

            for coluna in precos.columns:
                for h in range(horizonte):
                    ultimo_preco = preco_simul[coluna][j][-1]
                    
                    # Garantindo o acesso correto aos valores de kappa e media_lp
                    kappa_val = kappa[coluna].iloc[h] if isinstance(kappa[coluna], pd.Series) else kappa[coluna]
                    media_val = media_lp[coluna].iloc[h] if isinstance(media_lp[coluna], pd.Series) else media_lp[coluna][h]
                    
                    preco_simulado = (
                        ultimo_preco +
                        kappa_val * (media_val - ultimo_preco) +
                        volatilidade[coluna][h] * ret_corr[coluna].iloc[h] * ultimo_preco
                    )
                    preco_simul[coluna][j].append(preco_simulado)

                preco_simul[coluna][j].pop(0)
                montecarlo[coluna][j] = preco_simul[coluna][j]

        return montecarlo