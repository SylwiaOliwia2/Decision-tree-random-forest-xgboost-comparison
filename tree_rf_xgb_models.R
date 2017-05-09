#setwd("C:/Users/Sylwia/OneDrive/Licencjat_wlasciwe/House_prices")
rm(list=ls())
library(rpart) 
library(randomForest)
library(mice)
library(xgboost)
library(Matrix)
library(ggplot2)
library(reshape2)
library(caret)
source("funkcje.R")

#  Ladowanie danych ------------------------------------------------------------
     list.files()
     train <- read.csv("train.csv")
     test <- read.csv("test.csv")
     
     test_id <-test$Id
     test$Id <-NULL
     train$Id <-NULL

#  1. Models - based on raw data ------------------------------
     test$SalePrice <-0
     all_data <-rbind(train, test)
     
      set.seed(123)
     ind= sample(2,nrow(train),p=c(0.70,0.3),replace=T)
     cv = train[ind==2,]
     tr = train[ind==1,]
     xlab="real Price"
     ylab= "Predicted price"
     
     # tree
          tree = drzewo_decyzyjne(tr,cv) # cv prediction error 0.2402633
          save_submission(test,tree)   # kaggle prediction error  0.24030
          qplot(cv[,"SalePrice"],predict(tree, newdata=cv[,1:ncol(cv)-1]), xlab = xlab, ylab = ylab)
     
     # random forest
          # error due to NA's 
          #forest=lasy_losowe(tr,cv) 
          
     # XGBOOST
          dane_XGboost <- przygotuj_dane_XGBoost(all_data)
          xgboost_model_1 <-algorytm_XGBoost(dane_XGboost,nround=350,
                                             eta=0.3,
                                             verbose=0)   # cv error 0.157199
          qplot(cv[,"SalePrice"],xgboost_model_1$xgboost_pred, xlab = xlab, ylab = ylab)
          save_submission(dane_XGboost$test_X, xgboost_model_1$xgb) #  test set error on kaggle = 0.14366 

#  replace missing values with MICE (NA) -------------------------------------------
     test$SalePrice <-0
     all_data <-rbind(train, test)

     exclude <- c('PoolQC', 'MiscFeature', 'Alley', 'Fence')  # a lot of missing values
     include <- setdiff(names(all_data), exclude)
     # implemant MICE
     all_data <- all_data[include]
     imp.all_data <- mice(all_data, m=1, method='cart', printFlag=FALSE)
     all_data_complete <- complete(imp.all_data)
     
     any(is.na(all_data_complete)) # verify missing values
     # separate train and test set
     train_bez_NA<- all_data_complete[1:(nrow(train)),]
     test_bez_NA <-all_data_complete[(nrow(train)+1):(nrow(train)+nrow(test)),]

#  2. models with NA's filled in --------------------------
     cv_bez_NA = train_bez_NA[ind==2,]
     tr_bez_NA = train_bez_NA[ind==1,]
     
     # tree
          tree_bez_NA <- drzewo_decyzyjne(tr_bez_NA, cv_bez_NA)
          save_submission(test_bez_NA,tree_bez_NA)   # error on kaggle 0.24030
     
     #random_forest
     ntrees = seq(25,500,25)
     nodes= 2
     mtry = 9
     dobierz_ntree = dobierz_hiperparamery_lasy(tr_bez_NA,cv_bez_NA, nodes, mtry, ntrees)
     
     #grid serach
     nodes= seq(2,8,1)
     mtry = seq(5,100,10)
     ntrees = 200
     parametry_lasy = dobierz_hiperparamery_lasy(tr_bez_NA,cv_bez_NA, nodes, mtry, ntrees)
     lasy_bez_NA =lasy_losowe(tr_bez_NA,cv_bez_NA,
                              ntree=parametry_lasy$ntree, mtry=parametry_lasy$mtry, nodesize=parametry_lasy$node)
     save_submission(test_bez_NA, lasy_bez_NA$forest_model)
     
     #XGBOOST
          dane_XGboost_comp <- przygotuj_dane_XGBoost(all_data_complete)
          xgboost_model_bez_NA <- algorytm_XGBoost(dane_XGboost_comp ,all_data,nround=350,
                                                   eta=0.3,
                                                   verbose=0)  #cv error 0.1549436
          qplot(cv[,"SalePrice"],xgboost_model_bez_NA$xgboost_pred, xlab = "Cena rzeczywista", ylab = "Cena przewidziana")
          save_submission(dane_XGboost_comp$test_X,xgboost_model_bez_NA$xgb) # test error on kaggle 0.13876
     
#  feature engineering --------------------------
     all_data_complete$MSSubClass <- as.factor(all_data_complete$MSSubClass)
     newvals <- c("C (all)"="other","FV"="other","RL"="RL","RH"="resid","RM"= "resid")
     all_data_complete$MSZoning2 <- as.factor(newvals[as.character(all_data_complete$MSZoning)])
     
     all_data_complete$LotFrontage2 =log(log(all_data_complete$LotFrontage))
     all_data_complete$LotArea2 =log(log(all_data_complete$LotArea))
     all_data_complete$NotLotFrontage =log(log(all_data_complete$LotArea/all_data_complete$LotFrontage))
     ggplot(all_data_complete, aes(LotArea))+ geom_histogram()+xlab("Wartosc zmiennej LotArea")+
          ylab("Liczba wystapien")
     ggplot(all_data_complete, aes(LotArea2))+ geom_histogram()+xlab("Wartosc zmiennej LotArea")+
          ylab("Liczba wystapien")

     
#  3. desicion tree - choose best parameters --------------------------
     train_bez_NA<- all_data_complete[1:(nrow(train)),]
     test_bez_NA <-all_data_complete[(nrow(train)+1):(nrow(train)+nrow(test)),]

     cv_bez_NA = train_bez_NA[ind==2,]
     tr_bez_NA = train_bez_NA[ind==1,]
     
     tree_2 <- drzewo_decyzyjne(tr_bez_NA, cv_bez_NA)
     data_to_plot=as.data.frame(cbind(original_price=cv$SalePrice,y_pred=predict(tree_2, newdata=cv_bez_NA)))
     windows()
     ggplot(data_to_plot,aes(x=original_price,y=y_pred))+ geom_point()
     
     #split
     #Anova splitting has no parameters.
     
     #subset
     for (n in seq(0.4,1,0.1)){
          liczba_wierszy= nrow(tr_bez_NA)
          cat("Wartosc n=",n,".     ")
          tree_2<-drzewo_decyzyjne(tr_bez_NA, cv_bez_NA,subset=c(sample(1:liczba_wierszy,n*liczba_wierszy)))
          cat("\n")
     } #best for n= 1
     tree_2<-drzewo_decyzyjne(tr_bez_NA, cv_bez_NA)
     
     #prune
     tabelacp=printcp(tree_2) # cp = Complexity parameter
     tabelacp2= cbind(liczba_podzialow=tabelacp[,"nsplit"],cp_odchylenie_cp=tabelacp[,"xerror"] +tabelacp[,"xstd"])
     tabelacp2
     # choose parameters- description: http://stackoverflow.com/a/21397265/6401585
     # theoretically split should occure after n=11. let's try lower value
     error_init=0.4
     wart_i=0
     for (i in seq(0.3,0.005,-0.005)){
               #cat("cp=", i)
               przyceite_drzewo=prune.rpart(tree_2, cp=i)
               pred_tree <- predict(przyceite_drzewo, newdata=cv_bez_NA)
               error_tree= sqrt(sum((log(cv_bez_NA["SalePrice"]+1)-log(pred_tree+1))^2)/length(pred_tree))
               if (error_tree<error_init){
                    error_init=error_tree
                    wart_i=i
                    cat("Blad predykcji an danych walidacyjnych wynosi", error_init," dla i=",i)
                    cat("\n")
               }
     }
     # kaggle error the same as in previous examples
#  4. random forest - choose best parameters--------------------------
     # optimal amount of trees
     ntrees = seq(25,500,25)
     nodes= 4
     mtry = 75
     dobierz_ntree = dobierz_hiperparamery_lasy(tr_bez_NA,cv_bez_NA, nodes, mtry, ntrees)
     
     nodes = seq(2,8,1)
     mtry = seq(5,105,20)
     ntrees = 350
     #set up other parameters
     parametry_lasy = dobierz_hiperparamery_lasy(tr_bez_NA,cv_bez_NA, nodes, mtry, ntrees)
     lasy_bez_NA =lasy_losowe(tr_bez_NA,cv_bez_NA,
                              ntree=parametry_lasy$ntree, mtry=parametry_lasy$mtry, nodesize=parametry_lasy$node)
     save_submission(test_bez_NA, lasy_bez_NA$forest_model)
      
#  5.  XGBOOST - choose best parameters --------------------------
eta = seq(0.01,0.45,0.05)
liczba_drzew = seq(40,400,40)
subsamples   = seq(0.4,1,0.1)
child_weight = c(0.1,0.5,1,seq(2,20,2))
max_depth = seq(2,12,2)
lambda_alpha = c(0.001,0.01,seq(0.1,0.5,0.2),seq(0.6,1.2,0.1),1.5,2,2.5)

     dane_XGboost_featured <- przygotuj_dane_XGBoost(all_data_complete)
     
     param_list = dobierz_hipermaparametry_xgb(eta, liczba_drzew, subsamples, child_weight, max_depth, lambda_alpha)
     xgboost_model_final <-algorytm_XGBoost(dane_XGboost_featured,verbose=0,nround=param_list$nround,eta=param_list$eta,
                                            min_child_weight=param_list$min_child_weight, max_depth=param_list$max_depth,
                                            subsample=param_list$subsample, colsample_bytree=param_list$colsample_bytree,
                                            alpha=param_list$alpha, lambda=param_list$lambda)
     save_submission(dane_XGboost_featured$test_X, xgboost_model_final$xgb)
     
# #  6. computing time (dataset without NA's) ---------------------------------
n=10 # number of probes

### decisoin tree
time_tree <-system.time(replicate(n,tree_time()))
time_tree/n

### random Forest
time_forest <-system.time(replicate(n,forest_time()))
time_forest/n
(time_forest[2]+time_forest[1])/n
# highly depends of amount of trees
for (i in seq(250,450,100)){
     time_forest <-system.time(replicate(n,forest_time(ntree=i)))
     cat("Ntree = ", i)
     print(time_forest/n)
     cat((time_forest[2]+time_forest[1])/n,"\n")
} 

### xgboost
attach(dane_XGboost_comp)
time_xgb <-system.time(replicate(n,xgb_time()))
time_xgb/n
(time_xgb[2]+time_xgb[1])/n