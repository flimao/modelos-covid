# Modelos para previsão de casos confirmados de COVID com base em informações de cada paciente

<sub>Projeto para a disciplina de **Machine Learning** (Módulo 5) do Data Science Degree (turma de julho de 2020)</sub>

## Case

O objetivo do projeto será desenvolver um estudo em um *dataset* de COVID, base esta que contém informações sobre pacientes de COVID e se o paciente testou positivo ou não.

**A partir do diagnóstico de sintomas e informações dos pacientes deve-se desenvolver um modelo para prever casos confirmados de COVID.**

## Plano de trabalho

1. [__Preparação dos Dados e Verificação de Consistência__](notebooks_exploration/1-preproc.ipynb): Inicialmente faremos uma verificação da consistência dos dados e caso necessário efetuar eventuais modificações na base de dados. Alguns dos procedimentos que podemos fazer aqui são: Remoção e/ou tratamento de valores faltantes, remoção de duplicatas, ajustes dos tipos de variáveis, análise de _outliers_ entre outras;

2. [__Análise Exploratória dos Dados__](notebooks_exploration/2-eda.ipynb): Para fazermos a modelagem, precisamos conhecer muito bem os dados que estamos trabalhando. Por isso, nesta parte do projeto faremos análises e gráficos a respeito dos dados que estão utilizando;

3. [__Modelagem dos Dados__](notebooks_models/3-modelagem.ipynb): Nesta parte, vamos modelar um classificador para os resultados dos exames de COVID (campo `covid_res`). Vamos ajustar alguns modelos de acordo com alguma métrica de avaliação (a ser escolhida);

4. [__Otimização do Modelo__](notebooks_models/4-otimizacao.ipynb): A partir do modelo escolhido no tópico anterior, vamos tentar aprimorar e garantir um melhor desempenho no modelo, seja fazendo validação cruzada, otimização de parâmetros com `GridSearchCV` ou `RandomizedSearchCV` e até mesmo testar diferentes _thresholds_, ou seja, ao invés de utilizar a função `.predict` do modelo, vamos utilizar a função `.predict_proba` do modelo e a partir das probabilidades determinar qual vai ser o limiar onde será considerado um caso positivo ou negativo);

5. [__Conclusões sobre o Projeto__](notebooks_exploration/5-conclusao.ipynb): Para finalizar, vamos descrever nossas conclusões sobre o desenvolvimento do modelo e os resultados obtidos.

## *Dataset*

Caminho: [data/COVID.csv](data/COVID.csv)

A descrição das variáveis contidas no *dataset* pode ser encontradas a seguir:

- **`id`**: Identificação do paciente;
- **`sex`**: Sexo do Paciente (0 - Homem / 1 - Mulher);
- **`patient_type`**: Se o paciente foi dispensado para casa (1) ou se foi internado (0);
- **`intubed`**: Seo paciente foi intubado ou não;
- **`pneumonia`**: Se o paciente apresentou pneumonia ou não;
- **`age`**: Idade do Paciente;
- **`pregnancy`**: Se a paciente estava grávida ou não (para pacientes mulheres);
- **`diabetes`**: Se o paciente tem diabetes ou não;
- **`copd`**: Se o paciente tem COPD (*[chronic obstructive pulmonary disease](https://www.mayoclinic.org/diseases-conditions/copd/symptoms-causes/syc-20353679)*, mais comumente consequência de enfisema e/ou bronquite crônica) ou não;
- **`asthma`**: Se o paciente tem asma ou não;
- **`inmsupr`**: Se o paciente apresentou Imunosupressão ou não;
- **`hypertension`**: Se o paciente tem hipertensão ou não;
- **`other_disease`**: Se o paciente tem outras doenças ou não;
- **`cardiovascular`**: Se o paciente tem doenças cardiácas ou não;
- **`obesity`**: Se o paciente tem obesidade ou não;
- **`renal_chronic`**: Se o paciente tem problemas renais ou não;
- **`tobacco`**: Se o paciente é fumante ou não;
- **`contact_other_covid`**: Se o paciente teve contato com outras pessoas diagnosticadas com covid;
- **`icu`**: Se o paciente precisou ser internado na UTI; e
- **`covid_res`**: Se o resultado do teste foi Positivo ou Negativo.

----

## Conclusões

Resumimos aqui as principais conclusões alcançadas no trabalho de modelagem para predição de resultados de testes de COVID.

### Análise exploratória da base de dados

Antes do ajuste dos modelos, é importante conhecermos os dados: o que significa cada campo, o tipo dos dados (categóricos, numéricos, ordinais, texto etc), a ordem de grandeza dos valores, como eles se relacionam entre si etc.

Para tal, conduzimos uma análise exploratória dos dados de testes de COVID.

O único campo numérico é a idade; os outros são todos categóricos binários (homem/mulher, sim/não, positivo/negativo etc).
Embora haja tecnicamente 32.4 milhões de combinações dos valores únicos combinados de todos os campos, vimos que as colunas são altamente correlacionadas. 82% das correlações 2 a 2 foram consideradas estatisticamente significativas.
Eliminamos da análise os campos `intubed` e `icu` da análise, visto que nestes campos o preenchimento é muito inconsistente (acima de 70% de dados faltantes), de forma que em quaisquer novos provavelmente estes campos também não estarão preenchidos, o que os torna de pouco valor para o modelo.

Havia por volta de 50% de valores também no campo `pregnancy`, mas após a imposição de não-gravidez a homens, esses valores faltantes vieram a níveis razoáveis.

Eliminamos então os registros com valores faltantes do campo `pregnancy` e dos demais campos.
Vimos que, entre os pacientes que foram examinados e liberados para casa, não há muita diferença de idade entre os que tiveram um resultado positivo ou um negativo (entre 25 e 35 anos, aproximadamente). Já para os que foram internados, os pacientes que tiveram um resultado positivo de COVID são mais idosos e com dispersão muito menor que os com resultado negativo.
Inicialmente formulammos uma hipótese em que o paciente seria internado primariamente por presença de comorbidades evidentes, seja visualmente (*e.g.* obesidade) ou em exames básicos (*e.g.* pneumonia).

No entanto, essa hipótese não se confirmou; a presença de comorbidades responde por apenas 12% da correlação com o *status*  de internação.
Esperávamos que as comorbidades tivessem grande relação com o resultado do teste de COVID. Essa hipótese se confirmou: mesmo ajustando o nível de significância para considerar múlitplas comparações, chegamos aà conclusão que relação entre as comorbidades e o resultado do teste são estatisticamente significativas.

No entanto, essas relações são diferentes a depender da faixa etária:

- **Crianças até 10 anos**: estão mais vulneráveis ao acometimento por COVID se tem pneumonia ou problemas relacionados à obesidade (excesso de peso ou diabetes);
- **Adolescentes, adultos até 60 anos e idosos entre 60 e 80 anos**: estão mais vulneráveis ao contágio se tem qualquer comorbidade no conjunto de dados; e
- **Idosos com mais de 80 anos**: Estão mais vulneráveis ao acometimento por COVID se tem doenças pulmonares, se são imunossuprimidos ou se tem outras doenças.

No geral, as maiores comorbidades associadas a um teste positivo de COVID são pulmonares ou com pacientes imunossuprimidos.

Notamos também que,

- com exceção de crianças até 10 anos, **a prática regular do fumo está relacionada ao acometimento por COVID**; e
- em todas as faixas etárias, **a presença de problemas relacionados a obesidade (excesso de peso ou diabetes) está associada ao contágio por COVID**. A exceção são os muito idosos; uma razão para isso é o fato de os obesos não chegarem a idades mais avançadas em números suficientes.
A tabela abaixo mostra o teste de hipótese em que a hipótese nula $H_0$ é **não há nenhuma relação entre o grupo de comorbidades e o resultado do teste de COVID para cada faixa etária**.

Rejeitamos $H_0$?

|          Comorbidade | criança até 10 anos | adulto (entre 10 e 60 anos) | idoso (entre 60 e 80 anos) | muito idoso (acima de 80 anos) |
|----------------:|:-------------------:|:---------------------------:|:--------------------------:|:------------------------------:|
|         Coração |         Não         |             Sim             |             Sim            |               Não              |
|          Pulmão |         Sim         |             Sim             |             Sim            |               Sim              |
|       Obesidade |         Sim         |             Sim             |             Sim            |               Não              |
| Imunossupressão |         Não         |             Sim             |             Sim            |               Sim              |
|          Outras |         Não         |             Não             |             Sim            |               Sim              |

Uma outra variável potencialmente importante para o modelo é o sexo. Homens tem mais resultados positivos de COVID que mulheres (diferença estatisticamente significativa):

![testes chisq entre sexo e resultado teste de covid](assets/sexo_resultado_chisq.png)

Por fim, uma última variável que pode se mostrar relevante é se o paciente teve contato com outras pessoas comprovadamente com COVID (`contact_other_covid`). Ela tem relação com alguns outros campos.

### Construção do modelo

Após explorarmos bastante o conjunto de dados, partimos para a tentativa de construção de um modelo que capturasse nossos insights e produzisse uma predição adequada do resultado do teste de COVID com base nos campos.

Para tal, construímos uma `Pipeline` com dois segmentos:

1. Separação entre colunas numéricas (no caso, somente a idade) e categóricas (todo o restante).
   - para colunas numéricas, aplicamos uma normalização convertendo os valores em seus *z-scores*;
   - Para colunas categóricas, aplicamos um esquema de *one-hot-encoding*. Este passo só era estritamente necessário para o caso da coluna `contact_other_covid`, mas decidimos codificar o caso genérico, caso aparecessem no meio da análise outras variáveis que precisassem.

2. O modelo em si.

Para o modelo, testamos um total de 5 candidatos:

- **K Nearest Neighbors** (modelo baseado em distâncias);
- **Regressão logística** (modelo linear);
- **Decision Tree** (modelo baseado em árvores de decisão);
- **Random Forests** (modelo baseado em *bagging*); e
- **XGBoost** (modelo baseado em *boosting*).

Destes 5, a regressão logística e e o XGBoost eram claramente os melhores segundo a métrica AUROC:

![roc modelos default](assets/roc_default.png)
No entanto, mesmo sendo superior aos outros modelos, as métricas (em especial a acurácia) dos modelos não eram muito altas:

| Modelo              | AUROC | Brier Loss | Precisão | Recall | Acurácia |
|---------------------|-------|------------|----------|--------|----------|
| Regressão Logística | 65.6% | -0.2269    | 65.2%    | 38.7%  | 63.7%    |
| XGBoost             | 64.8% | -0.230     | 63.2%    | 40.1%  | 63.1%    |

#### Análise de importância dos atributos

Antes de prosseguirmos à otimização, uma análise de importância relativa de cada *feature* do modelo foi feita.
![importância relativa dos features](assets/shap-importance.png)
O campo mais importante para os dois modelos finalistas foi a idade, seguida de se o paciente tem ou não pneumonia, se o paciente teve contato com outras pessoas com COVID, se o paciente foi internado ou liberado para casa, se era homem ou mulher e se era obeso.

Isso está em linha com o que suspeitávamos após a análise exploratória da base.
Análisamos também o impacto das *features na resposta final com base em seus valores:

![impacto das features em relação a seus valores](assets/shap-magnitude.png)
Vemos que o modelo de regressão logística captura pouco das interações entre as características dos pacientes:

- a influência dos fatores está dividida claramente conforme seus valores; e
- dentre os casos em que um *feature* influi bastante no modelo, essa influência é sempre a mesma.

Já o modelo *XGBoost* captura as interações muito bem: há no gráfico de *swarm* muitos valores intermediários com influência, evidenciando que há interações relevantes no modelo. A quantificação da influência (representada pelos valores de Shapley) está mais distribuída entre os casos em que uma *feature* tem influência no modelo.

Confirmando o que vimos no gráfico de barra, alguns dos fatores que menos influenciam a resposta do modelo (`renal_chronic`, `asthma`, `inmsupr`, `copd`, `cardiovascular`) nos dois modelos são campos com pouca prevalência da classe positiva. Não é surpreendente que tenham pouca influência.

Nos dois modelos, o fator que mais teve influência na previsão de testes de COVID foi a idade. Pessoas mais idosas tendem o modelo para prever teste positivo mais frequentemente. No modelo *XGBoost*, há bastante interação da idade com outros fatores.

Alguns fatores que tendem o modelo fortemente a prever testes positivos:

- Pneumonia;
- Internação;
- Paciente do sexo masculino;
- Obesidade ou diabetes; e
- Paciente não-fumante!

A previsão do resultado do teste para COVID tende para o lado positivo em pacientes cujo contato com outras pessoas que tenham COVID não tenha sido registrado.

Surpreendentemente, pacientes fumantes ou com problemas cardiovasculares tendem o modelo a prever testes negativos, mas essa tendência não é muito forte.
Um detalhe interessante é que, entre os campos em que há havia baixa prevalência da classe positiva, todas as ocorrências da classe positiva correspondem a uma grande influência na resposta do modelo:

![impacto shap de features com baixa prevalencia](assets/shap-baixa-prevalencia.png)
O próximo passo foi então otimizar o modelo de forma a melhorar um pouco estas métricas.

### Otimização dos modelos

Cada modelo de aprendizado de máquina é construído em cima de alguns parâmetros. Por exemplo, no modelo de regressão logística, há um parâmetro regulando a presença ou não do intercepto (`fit_intercept`) e um outro controlando a regularização do modelo, *i.e.* a penalização para muitos coeficientes (`penalty`), entre outros. Estes são chamados ***hiperparâmetros***.

A otimização do modelo consiste na escolha dos hiperparâmetros de forma a maximizar/minimizar alguma métrica. Neste caso, para sermos consistentes com o estudo realizado na parte de modelagem, vamos testar várias combinações de hiperparâmetros e ranqueá-las através do *Brier score*. Um modelo se adere mais ao resultado se essa métrica for menor que a de um outro modelo; logo, minimizamos o *Brier score* das combinações de hiperparâmetros. Levamos em conta também a área sob a curva ROC.
Dos modelos de aprendizado de maquina que testamos para predizer o resultado dos testes de COVID, dois performaram muito bem com parâmetros *default*: regressão logística e *XGBoost*.

Para cada um deles, realizamos uma busca no espaço de hiperparâmetros para determinar quais eram mais aplicáveis ao problema em questão, *i.e.* qual conjunto performaria melhor nas métricas escolhidas (*Brier score* e AUROC).
Como podemos ver na tabela abaixo, o modelo de regressão logística começou com métricas ligeiramente melhores, mas com parâmetros otimizados o modelo XGBoost performou melhor:

| Modelo                        | AUROC | Brier Loss | Precisão | Recall | Acurácia |
|-------------------------------|-------|------------|----------|--------|----------|
| Regressão Logística           | 65.2% | -0.2269    | 65.2%    | 38.7%  | 63.7%    |
| Regressão Logística otimizado | 65.2% | -0.2269    | 65.4%    | 38.2%  | 63.8%    |
| XGBoost                       | 64.9% | -0.2300    | 63.2%    | 40.1%  | 63.1%    |
| XGBoost otimizado             | 66.0% | -0.2289    | 66.0%    | 38.4%  | 64.0%    |

![curva roc otimizada](assets/roc_otimizado.png)
Por esse motivo, e pelo fato de o modelo XGBoost capturar melhor as interrelações entre os campos,o escolhemos com os seguintes parâmetros:

- `base_score`: 0.5,
- `booster`: 'gbtree',
- `colsample_bylevel`: 1,
- `colsample_bynode`: 1,
- `colsample_bytree`: 1,
- `criterion`: 'mae',
- `enable_categorical`: False,
- `eval_metric`: 'logloss',
- `gamma`: 0,
- `gpu_id`: -1,
- `importance_type`: None,
- `interaction_constraints`: '',
- `learning_rate`: 0.1,
- `max_delta_step`: 0,
- `max_depth`: 3,
- `max_features`: 'log2',
- `min_child_weight`: 1,
- `missing`: nan,
- `monotone_constraints`: '()',
- `n_estimators`: 100,
- `n_jobs`: 8,
- `num_parallel_tree`: 1,
- `objective`: 'binary:logistic',
- `predictor`: 'auto',
- `random_state`: 42,
- `reg_alpha`: 0,
- `reg_lambda`: 1,
- `scale_pos_weight`: 1,
- `subsample`: 1,
- `tree_method`: 'exact',
- `use_label_encoder`: True,
- `validate_parameters`: 1,
- `verbosity`: None
Conduzimos ainda uma análise de otimização do threshold de decisão. Entendemos que um falso negativo na predição de um resultado de teste de COVID (predizer que o resultado dará negativo, mas o resultado dar positivo) é mais prejudicial que um falso positivo; no primeiro caso, o paciente será liberado e pode infectar outras pessoas, facilitando a transmissão; já no segundo, o paciente se isolará sem necessidade, porém sem maiores consequências.

Em outras palavras, a métrica *recall* (dentre os resultados de testes positivos/negativos, quantos acertamos?) é mais importante para a otimização do *threshold* que a precisão (dentre os resultamos que predizemos ser positivos, quantos acertamos?).

Sendo assim, utilizamos duas métricas de referência:

- **Custo estimado**: estimamos que o custo de um falso negativo é 5x maior que o custo de um falso positivo; e
- **F-$\beta$ score**: como priorizamos o *recall* em detrimento da precisão, utilizamos um $\beta = 1.5$

Conforme observamos no gráfico...

![otimizacao de threshold](assets/custo_threshold.png)

...um *threshold* $\theta = 24\%$ parece ser o melhor para limiar de tomada de decisão.
Com esse threshold otimizado, o modelo passa a ter as seguites métricas:

| Métrica  | Valor |
|:----------:|:-------:|
| Acurácia | $45.5\%$ |
| F1       | $61.6\%$ |
| Precisão | $44.7\%$ |
| Recall   | $99.0\%$ |

### Próximos passos

Estimamos que uma grande melhoria que pode ser feita ao modelo é o chamado ***feature engineering***, onde combinamos valores de cada campo e construímos um outro campo derivado. Este novo campo pode aumentar a acurácia do modelo significativamente. Iniciamos o processo de *feature engineering* na análise exploratória de dados, ao correlacionar o resultado do teste de COVID a grupos de comorbidades (cardíacas, pulmonares etc).

Uma outra melhoria que poderia ser feita ao modelo é refinar a matriz de custos estimada na otimização do *threshold* de decisão. Por exemplo, poderia ser feito um pequeno estudo sobre as taxas de mortalidade de COVID e perda de produtividade da população brasileira vs o impacto do *lockdown* na vida das pessoas.
