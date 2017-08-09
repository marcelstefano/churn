library(ggplot2)
library(caret)
library(doMC)
library(dplyr)
library(cowplot)
registerDoMC(cores=2)

url<-"https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df<-read_csv(url)

df<-df %>% map_df(~ifelse(str_detect(.x,"No"),"No",.x))

df<-na.omit(df)



df<-as.data.frame(unclass(df))
df$tenure_interval<-cut(df$tenure,breaks=c(0,6,12,24,36,48,62,72),labels=c("0 - 6 months","6 - 12 months","12-24 months","24-36 months","36-48 months","48-62 months","> 62 months"))

## Extrair somente o data.frame e as variáveis de interesse.





## Divisão da base de dados em training e testing para análise e validação posterior

btrain<-createDataPartition(df$Churn,p=.80,list=F)

train<-df[btrain,-c(1,6)]
test<-df[-btrain,-c(1,6)]

## Estabele os parámetros para o trainamento do modelo.

ctrl <- trainControl(method = "repeatedcv", # Para resampling usa validação cruzada repetica
                     number = 10, ## Número de iterações
                     repeats = 5, ## Número de folds a serem computados
                     summaryFunction = twoClassSummary, ## Função para computar métricas de desempenho na validação cruzada
                     classProbs = TRUE, ## Computa as probabilidades das classes/etiquetas
                     savePredictions = TRUE, ## salva as predições no resampling
                     allowParallel = TRUE, ## autoriza paralelização.
                     sampling="up" ## Equilibra as classes para cima, já que a maioria é "denegado"
)


## Regressão logística

mod_GLM <- train(Churn ~ .,data=train, method="glm", family="binomial",
                 trControl = ctrl, tuneLength = 5,
                 metric = "ROC")

#### Predição

pglm<-predict(mod_GLM,test)
ppglm<-confusionMatrix(pglm,test$Churn)
ppglm


#### Diagnóstico

p<-predict(mod_GLM,test,"prob")[[1]]
t<-ifelse(test$decisao=="concedido",1,0)
a<-classifierplots(t,p)

## Gradient boosting

# Uma vez que a base não é muito grande, preferimos utilizar um encolhimento de 0.001 e 0.01.
# A base final optou por 0.01. Então este foi usado.

grid_gbm <- expand.grid(interaction.depth=5, n.trees = 250,
                        shrinkage=0.01,
                        n.minobsinnode=10)

mod_GBM <- train(train[21] ~ train[1:20,22],  data=train, method="gbm",
                 trControl = ctrl,tuneGrid=grid_gbm,tuneLength = 5,metric = "ROC")

#### Predição

pgbm<-predict(mod_GBM,test)
ppgbm<-confusionMatrix(pgbm,test$Churn)
ppgbm




# O modelo a seguir utiliza C5.0 com boosting

grid_c50 <- expand.grid( .winnow = TRUE, .trials=30, .model="rules" )

mod_C50 <- train(Churn~.,data=train, method = "C5.0",
                 trControl = ctrl,tuneGrid=grid_c50,tuneLength = 5,metric = "ROC")

pC50<-predict(mod_C50,test)
ppC50<-confusionMatrix(pC50,test$Churn)
ppC50


## Floresta aleatória
grid_rf <- expand.grid(.mtry=2)

mod_RF <- train(y=make.names(train$Churn),x=train[c(1:18,20)],method = "ranger",
                trControl = ctrl,tuneLength = 5,metric = "ROC", num.threads = 4,importance="impurity")

pRF<-predict(mod_RF,test)

ppRF<-confusionMatrix(pRF,test$Churn)
ppRF


## XGBoost

grid_XGB <- expand.grid(nrounds = 100,
                        max_depth = 1,
                        eta = .3,
                        gamma = 0,
                        colsample_bytree = .6,
                        min_child_weight = 1,
                        subsample = 1)

mod_XGB <- train(Churn~.,data=train, method = "xgbTree",
                 trControl = ctrl,metric = "ROC",tune.Grid=grid_XGB)

pXGB<-predict(mod_XGB,test)
ppXGB<-confusionMatrix(pXGB,test$Churn)
ppXGB


### Comparando os modelos

results <- resamples(list(GLM=mod_GLM,
                          GBM=mod_GBM,
                          C50=mod_C50,
                          RF=mod_RF,
                          XGB=mod_XGB),decreasing=T,
                     
                     metrics="ROC")

summary(results)

## Plotando a importância das variáveis:

## Cria uma lista com todas os modelos

mod_lista<-list(mod_GLM,mod_C50,mod_RF,mod_XGB)

## Aplica o ggplot a esta lista

a<-mod_lista %>% map(~ggplot(varImp(.x))+labs(x="Variáveis", y="Importância"))

## Coloca todo mundo num grid:

plot_grid(plotlist = a,labels=c("GLM","C5.0","RF","XGB"))


## Explicando regressão logística

explain <- lime(train, mod_C50)


pred <- data.frame(sample_id = 1:nrow(test),
                   predict(mod_C50, test, type = "prob"),
                   observado = test$Churn)
pred$predito <- colnames(pred)[2:4][apply(pred[, 2:4], 1, which.max)]

pred$correto <- ifelse(pred$observado == pred$predito, "correto", "errado")


pred_correto <- filter(pred, correto == "correto")
pred_errado <- filter(pred, correto == "errado")

test_data_correto <- test %>%
  mutate(sample_id = 1:nrow(test)) %>%
  filter(sample_id %in% pred_correto$sample_id) %>%
  sample_n(size = 2) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id") %>%
  select(-Churn)

explanation_cor <- explain(test_data_correto, n_labels = 2, n_features = 20)

plot_features(explanation_cor, ncol = 2)+
  labs(x="Variável",y="Peso")+
  scale_fill_manual(values=c("darkgreen","darkred"),
                    labels=c("apoia","contraria"))

g1<-ggplot_build(gg)

ggsave(filename = "gg_certo.pdf",width=15,height=7,device = cairo_pdf)


test_data_errado <- test %>%
  mutate(sample_id = 1:nrow(test)) %>%
  filter(sample_id %in% pred_errado$sample_id) %>%
  sample_n(size = 2) %>%
  remove_rownames() %>%
  tibble::column_to_rownames(var = "sample_id")


explanation_errado <- explain(test_data_errado, n_labels = 2, n_features = 8)


plot_features(explanation_errado, ncol = 2)
labs(x="Variável",y="Peso")+
  scale_fill_manual(values=c("darkgreen","darkred"),
                    labels=c("apoia","contraria"))

#ggsave(filename = "~/R/custodia/plots/gg_errado.pdf",width=15,height=7,device = cairo_pdf)