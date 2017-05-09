# save test prediction in Kaggle format
save_submission <- function(dataset, model) {
     Id = as.data.frame(test_id)
     SalePrice <- as.data.frame(predict(model, newdata=dataset))
     submission = cbind(Id, SalePrice)
     colnames(submission) <- c("Id","SalePrice")
     write.csv(submission, file="submission.csv", row.names=FALSE)
}

# decision tree - return tree model
drzewo_decyzyjne <- function(train_data, cv_data, ...){
     set.seed(123)   
     tree <- rpart(SalePrice ~ ., method="anova", data=train_data, ...)
     pred_tree <- predict(tree, newdata=cv_data)
     error_tree= sqrt(sum((log(cv_data["SalePrice"]+1)-log(pred_tree+1))^2)/length(pred_tree))
     cat("Blad predykcji an danych walidacyjnych wynosi", error_tree)
     return (tree)
}

#random forest - return model, cv prediction, cv error
lasy_losowe <- function(train_data, cv_data, ...){
     set.seed(123)
     forest <- randomForest(x=train_data[,!(names(train_data) %in% "SalePrice")],
                            y=train_data[,"SalePrice"], na.action=rfImpute,...)
     pred_forest <- predict(forest,newdata=cv_data)
     error_forest= sqrt(sum((log(cv_data["SalePrice"]+1)-log(pred_forest+1))^2)/length(pred_forest))
     return (list("forest_model"=forest, "predykcjca"=pred_forest, "error_forest"=error_forest))
}

# fine-tune parameters for Random Forest.
# domain 'train' function tune onlu mtry parameter
# return list of optimal parameters
dobierz_hiperparamery_lasy <- function(tr_bez_NA,cv_bez_NA, nodes, mtry, ntrees){
     param_list=list()
     min_error=1000
     for (node in nodes){
          for (mtr in mtry){
               for (ntre in ntrees){
                    lasy_bez_NA =lasy_losowe(tr_bez_NA,cv_bez_NA, ntree=ntre,mtry=mtr, nodesize=node)
                    if (lasy_bez_NA$error_forest < min_error){
                         min_error = lasy_bez_NA$error_forest
                         param_list$node=node
                         param_list$mtry = mtr
                         param_list$ntree= ntre
                         cat("\nNowe parametry ustawione: node: ",param_list$node,", mtry: ",param_list$mtry, 
                             ", ntree: ", param_list$ntree, ", error: ",min_error )
                    }
               }}}
     return(param_list)
     }


# prepare dataset to xgboost
# return X and Y (saparated for train, cv, test )
przygotuj_dane_XGBoost <- function(wszystkie_dane){
     previous_na_action <- options('na.action')
     options(na.action='na.pass')
     X =sparse.model.matrix(SalePrice~.-1, data = wszystkie_dane) # convert data_frame to sparse_matrix
     options(na.action=previous_na_action$na.action)
     train_y=(as.matrix(train["SalePrice"]))
     train_X = X[1:1460,]
     test_X = X[1461:nrow(wszystkie_dane),]
     
     set.seed(123)
     spl= sample(2,nrow(train),p=c(0.70,0.3),replace=T)
     tr_X<-train_X[spl==1,]
     cv_X<-train_X[spl==2,]
     tr_y<-(train_y[spl==1,])
     cv_y<-(train_y[spl==2,])
     return( list("tr_X"=tr_X, "cv_X"=cv_X, "tr_y"=tr_y, "cv_y"=cv_y, "test_X"=test_X))
}

# xgboost model- for linear regression
algorytm_XGBoost <-function(lista_danych, ...){
     attach(lista_danych)
     set.seed(123)
     xgb <- xgboost(data = tr_X, 
                    label = tr_y,
                    eval_metric = "rmse",  
                    objective = "reg:linear", 
                    ...
     )
     
     xgboost_pred <- predict(xgb, cv_X)
     error_xgb= sqrt(sum((log(cv_y+1)-log(xgboost_pred+1))^2)/length(cv_y))
     return(list("error_xgb"=error_xgb,"xgb"=xgb, "xgboost_pred"=xgboost_pred))
}

# fine - tune XGBoost model
# FUNCTION NEED MORE ELEGANT FORM!
dobierz_hipermaparametry_xgb <-function(eta, liczba_drzew, subsamples, child_weight, max_depth, lambda_alpha){
     param_list = list()
     min_error = 1000  #higher than any previous validation error 
     # eta, number of trees
     for (d in liczba_drzew){
          for (e in eta){
               xgboost_model_final <-algorytm_XGBoost(dane_XGboost_featured,verbose=0,
                                                      nround=d,
                                                      eta=e) 
               if (xgboost_model_final$error_xgb < min_error){
                    min_error = xgboost_model_final$error_xgb
                    param_list$eta=e
                    param_list$nround = d
                    cat("New minimal error:", min_error)
               }}}
     # max_depth, min_child_weight
     for (max_d in max_depth){
          for (child_w in child_weight){
               
               xgboost_model_final <-algorytm_XGBoost(dane_XGboost_featured,verbose=0,nround=param_list$nround,eta=param_list$eta,
                                                      min_child_weight=child_w, max_depth=max_d
               ) 
               if (xgboost_model_final$error_xgb < min_error){
                    min_error = xgboost_model_final$error_xgb
                    param_list$min_child_weight=child_w
                    param_list$max_depth = max_d
               }}}
     # subsample, colsample
     for (sub in subsamples){
          for (col in subsamples){
               
               xgboost_model_final <-algorytm_XGBoost(dane_XGboost_featured,verbose=0,nround=param_list$nround,eta=param_list$eta,
                                                      min_child_weight=param_list$min_child_weight, max_depth=param_list$max_depth,
                                                      subsample=sub, colsample_bytree=col
               ) 
               if (xgboost_model_final$error_xgb < min_error){
                    min_error = xgboost_model_final$error_xgb
                    param_list$subsample=sub
                    param_list$colsample_bytree = col
               }}}
     # alpha, lambda
     for (a in lambda_alpha){
          for (l in lambda_alpha){
               
               xgboost_model_final <-algorytm_XGBoost(dane_XGboost_featured,verbose=0,nround=param_list$nround,eta=param_list$eta,
                                                      min_child_weight=param_list$min_child_weight, max_depth=param_list$max_depth,
                                                      subsample=param_list$subsample, colsample_bytree=param_list$colsample_bytree,
                                                      alpha=a, lambda=l
               ) 
               if (xgboost_model_final$error_xgb < min_error){
                    min_error = xgboost_model_final$error_xgb
                    param_list$alpha=a
                    param_list$lambda=l
               }}}
     # eta once again
     for (e in eta){
          xgboost_model_final <-algorytm_XGBoost(dane_XGboost_featured,verbose=0,nround=param_list$nround,
                                                 min_child_weight=param_list$min_child_weight, max_depth=param_list$max_depth,
                                                 subsample=param_list$subsample, colsample_bytree=param_list$colsample_bytree,
                                                 alpha=param_list$alpha, lambda=param_list$lambda, eta=e
          ) 
          if (xgboost_model_final$error_xgb < min_error){
               min_error = xgboost_model_final$error_xgb
               param_list$eta=e
          }}
     return(param_list) 
}

### PERFORMANCE TIME
tree_time<-function(){
     tree <- rpart(SalePrice ~ ., method="anova", data=tr_bez_NA)
     pred_tree <- predict(tree, newdata=cv_bez_NA)
}

forest_time<-function(ntree=500){
     forest <- randomForest(x=tr_bez_NA[,!(names(tr_bez_NA) %in% "SalePrice")],
                            y=tr_bez_NA[,"SalePrice"], na.action=rfImpute, ntree = ntree)
     pred_forest <- predict(forest,newdata=cv_bez_NA)
}

xgb_time<-function(){
     xgb <- xgboost(data = tr_X, 
                    label = tr_y,
                    eval_metric = "rmse",  
                    objective = "reg:linear",
                    nrounds = 150
     )
     xgboost_pred <- predict(xgb, cv_X)
}