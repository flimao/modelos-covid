# Modelos para previsão de casos confirmados de COVID com base em informações de cada paciente

<sub>Projeto para a disciplina de **Machine Learning** (Módulo 5) do Data Science Degree (turma de julho de 2020)</sub>

## Case

O objetivo do projeto será desenvolver um estudo em um *dataset* de COVID, base esta que contém informações sobre pacientes de COVID e se o paciente testou positivo ou não.

**A partir do diagnóstico de sintomas e informações dos pacientes deve-se desenvolver um modelo para prever casos confirmados de COVID.**

## Plano de trabalho

1. [__Preparação dos Dados e Verificação de Consistência__](notebooks_exploration/1-preproc.ipynb): Inicialmente faremos uma verificação da consistência dos dados e caso necessário efetuar eventuais modificações na base de dados. Alguns dos procedimentos que podemos fazer aqui são: Remoção e/ou tratamento de valores faltantes, remoção de duplicatas, ajustes dos tipos de variáveis, análise de _outliers_ entre outras;

2. [__Análise Exploratória dos Dados__](notebooks_exploration/2-eda.ipynb): Para fazermos a modelagem, precisamos conhecer muito bem os dados que estamos trabalhando. Por isso, nesta parte do projeto faremos análises e gráficos a respeito dos dados que estão utilizando;

3. [__Modelagem dos Dados__](notebooks_exploration/3-modelagem.ipynb): Nesta parte, vamos modelar um classificador para os resultados dos exames de COVID (campo `covid_res`). Vamos ajustar alguns modelos de acordo com alguma métrica de avaliação (a ser escolhida);

4. [__Otimização do Modelo__](notebooks_exploration/4-otimizacao.ipynb): A partir do modelo escolhido no tópico anterior, vamos tentar aprimorar e garantir um melhor desempenho no modelo, seja fazendo validação cruzada, otimização de parâmetros com `GridSearchCV` ou `RandomizedSearchCV` e até mesmo testar diferentes _thresholds_, ou seja, ao invés de utilizar a função `.predict` do modelo, vamos utilizar a função `.predict_proba` do modelo e a partir das probabilidades determinar qual vai ser o limiar onde será considerado um caso positivo ou negativo);

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
