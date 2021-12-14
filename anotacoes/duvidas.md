# Dúvidas para monitoria

1. Desbalanceio em um campo da variável independentes não auxilia muito, mas chega a atrapalhar?
    * mencionar comorbidades (`inmsupr` tem 1.6% positivos)
        * `inmsupr`
        * `copd`
        * `renal_chronic`
        * `cardiovascular`
        * `other_disease`
        * `asthma`
    * **RESPOSTA**: não chega a atrapalhar, e pode até ajudar um pouco. Com uma base desse tamanho, não é um problema deixar. Mas por política, campos com ~30% dos dados faltando ou mais são descartados.

2. É válido codificar os `NaN` de um campo como uma outra categoria?
    * mencionar `contact_other_covid` (30% `NaN`)
    * pode ser que esteja MNAR (*missing not at random*):
    * pode ser que, por exemplo, o registro dependa do local onde foi realizado o teste. Um local de teste sem monitoramento de contato pode representar uma variabilidade maior, ou instalações mais precárias, o que pode enviesar o resultado do teste.

    * **RESPOSTA**: sim é válido.

3. É válido usar AUROC para comparar passos de pré-processamento?
    * mencionar passos alternativos (passos para o campo `contact_other_covid` e dropar ou não as colunas com prevalência baixa)
    * **RESPOSTA**: sim é válido

4. Faz sentido descartar duplicatas?
    * **RESPOSTA**: Nesse caso não pois os `id`s são diferentes. Mesmo que não fosse, não faria sentido pois essa é a forma segundo a qual os dados foram coletados.

5. Há no `scikit-learn` uma medida de correlação entre variáveis categóricas?
    * **RESPOSTA**: Cramer V (***)
    * Theil U
