data = read.csv("C:/Users/oweng/OneDrive - Youngstown State University/Summer 2026/Predictive Modeling Algorithms/FinalProject-OwenPfeifer/Car_Insurance_Claim.csv")

library(dplyr)
library(readr)
library(tidyr)
library(tree)
library(glmnet)
library(boot)
library(randomForest)
library(ggplot2)
library(pROC)

# data cleaning and formatting
colnames(data)

colSums(is.na(data))

data_clean = data %>%
  mutate(
    AGE = case_when(
      AGE == "16-25" ~ 0,
      AGE == "26-39" ~ 1,
      AGE == "40-64" ~ 2,
      AGE == "65+" ~ 3
    ),
    DRIVING_EXPERIENCE = case_when(
      DRIVING_EXPERIENCE == "0-9y"   ~ 0,
      DRIVING_EXPERIENCE == "10-19y" ~ 1,
      DRIVING_EXPERIENCE == "20-29y" ~ 2,
      DRIVING_EXPERIENCE == "30y+"   ~ 3
    ),
    EDUCATION = case_when(
      EDUCATION == "none"        ~ 0,
      EDUCATION == "high school" ~ 1,
      EDUCATION == "university"  ~ 2
    ),
    INCOME = case_when(
      INCOME == "poverty"       ~ 0,
      INCOME == "working class" ~ 1,
      INCOME == "middle class"  ~ 2,
      INCOME == "upper class"   ~ 3
    ),
    GENDER = ifelse(GENDER == "male", 1, 0),
    RACE = ifelse(RACE == "majority", 1, 0),
    VEHICLE_YEAR = ifelse(VEHICLE_YEAR == "after 2015", 1, 0),
    VEHICLE_TYPE = ifelse(VEHICLE_TYPE == "sports car", 1, 0),
    
  ) %>%
  select(-ID,-POSTAL_CODE) %>%
  drop_na(CREDIT_SCORE, ANNUAL_MILEAGE)

# Exploratory Analysis
summary(data_clean)

ggplot(data_clean, aes(x = AGE, fill = as.factor(OUTCOME))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "salmon"), 
                    labels = c("0" = "No Claim", "1" = "Claim Filed")) +
  labs(
    title = "Proportion of Insurance Claims by Age Group",
    x = "Age Group(16-25 - 0,26-39 - 1,40-64 - 2,65+ - 3)",
    y = "Percentage of Drivers",
    fill = "Outcome"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data_clean, aes(x = DRIVING_EXPERIENCE, fill = as.factor(OUTCOME))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "salmon"), 
                    labels = c("0" = "No Claim", "1" = "Claim Filed")) +
  labs(
    title = "Proportion of Insurance Claims by Driving Experience",
    x = "Years of Experience(0-9y - 0,10-19 - 1,20-29y - 2, 30y+ - 3)",
    y = "Percentage of Drivers",
    fill = "Outcome"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data_clean, aes(x = GENDER, fill = as.factor(OUTCOME))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("0" = "lightblue", "1" = "salmon"), 
                    labels = c("0" = "No Claim", "1" = "Claim Filed")) +
  labs(
    title = "Proportion of Insurance Claims by Gender",
    x = "Gender(Female - 0, Male - 1)",
    y = "Percentage of Drivers",
    fill = "Outcome"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data_clean, aes(x = AGE)) +
  geom_bar(fill = "lightblue") +
  labs(
    title = "Distribution of Policyholders by Age Group",
    x = "Age Group(16-25 - 0,26-39 - 1,40-64 - 2,65+ - 3)",
    y = "Total Number of Policyholders"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data_clean, aes(x = DRIVING_EXPERIENCE)) +
  geom_bar(fill = "lightblue") +
  labs(
    title = "Distribution of Policyholders by Driving Experience",
    x = "Driving Experience(0-9y - 0,10-19 - 1,20-29y - 2, 30y+ - 3)",
    y = "Total Number of Policyholders"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggplot(data_clean, aes(x = GENDER)) +
  geom_bar(fill = "lightblue") +
  labs(
    title = "Distribution of Policyholders by Gender",
    x = "Gender(Female - 0, Male - 1)",
    y = "Total Number of Policyholders"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# creating train and test samples
set.seed(1)
n = nrow(data_clean)
train.i = sample(1:n, size = 0.7 * n)

train_data = data_clean[train.i,]
test_data = data_clean[-train.i,]

# logistic model training
logreg = glm(OUTCOME ~ ., data = train_data, family = "binomial")

summary(logreg)

log_pred = predict(logreg,newdata = test_data, type = "response")

predicted_classes <- ifelse(logprediction > 0.5, 1, 0)

conf_table <- table(Predicted = predicted_classes, Actual = test_data$OUTCOME)

print(conf_table)

# Simple decision tree model
class_tree = tree(OUTCOME ~ ., data = train_data)
summary(class_tree)

tree_pred = predict(class_tree, newdata = test_data)

plot(class_tree)
text(class_tree, pretty = 0)

tree_pred_class <- ifelse(tree_pred > 0.5, 1, 0)

conf_table_tree <- table(Predicted = tree_pred_class, Actual = test_data$OUTCOME)
print(conf_table_tree)

#improved logistic regression model based off p value with cross validation 

improved_logreg = glm(OUTCOME ~ GENDER + DRIVING_EXPERIENCE + 
                      VEHICLE_OWNERSHIP + VEHICLE_YEAR + MARRIED
                      + CHILDREN + ANNUAL_MILEAGE + PAST_ACCIDENTS,
                      data = data_clean,
                      family = "binomial")
summary(improved_logreg)

improved_logreg_pred = predict(improved_logreg, newdata = test_data, type = "response")

improved_log_pred_class <- ifelse(improved_logreg_pred > 0.5, 1, 0)


conf_table_improved_log <- table(Predicted = improved_log_pred_class, Actual = test_data$OUTCOME)
print(conf_table_improved_log)

cost_funct <- function(actual, predicted) {
  mean(abs(actual - predicted) > 0.5) 
}

set.seed(1)
cv_results = cv.glm(data = data_clean,
                    glmfit = improved_logreg,
                    cost = cost_funct,
                    K = 10)

cv_error_rate <- cv_results$delta[1]
print(paste("Cross-Validated Accuracy:", round(1 - cv_error_rate, 4)))

# lasso method to find important predictors
x_train = model.matrix(OUTCOME ~ ., data = train_data)[, -1]
y_train = train_data$OUTCOME

x_test = model.matrix(OUTCOME ~ ., data = test_data)[, -1]
y_test = test_data$OUTCOME

set.seed(1)
lasso = cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")

lasso_lambda = lasso$lambda.min
cat("Optimal Lambda:", lasso_lambda)

lasso_pred = predict(lasso, s = lasso_lambda, newx = x_test, type = "response")
  
lasso_pred_class = ifelse(lasso_pred > 0.5, 1, 0)                   

conf_table_lasso <- table(Predicted = lasso_pred_class, Actual = y_test)
print(conf_table_lasso)

lasso_coef = predict(lasso, type = "coefficients", s = lasso_lambda)
coef_matrix = as.matrix(lasso_coef)

print(coef_matrix)

# random forest
train_data$OUTCOME <- as.factor(train_data$OUTCOME)
test_data$OUTCOME <- as.factor(test_data$OUTCOME)

set.seed(1)
rf_model <- randomForest(OUTCOME ~ ., data = train_data, importance = TRUE, ntree = 500)

print(rf_model)

rf_pred <- predict(rf_model, newdata = test_data, type = "prob")[, 2]

rf_pred_class <- ifelse(rf_pred > 0.5, 1, 0)

conf_table_rf <- table(Predicted = rf_pred_class, Actual = test_data$OUTCOME)
print(conf_table_rf)

# Model Evaluation
roc_basic_log    <- roc(test_data$OUTCOME, log_pred)
roc_tree         <- roc(test_data$OUTCOME, tree_pred)
roc_improved_log <- roc(test_data$OUTCOME, improved_logreg_pred)
roc_lasso        <- roc(test_data$OUTCOME, as.numeric(lasso_pred))
roc_rf           <- roc(test_data$OUTCOME, as.numeric(rf_pred))

plot(roc_rf, col = "red", lwd = 2, main = "Combined ROC Curve - Model Comparison")
plot(roc_lasso, col = "blue", lwd = 2, add = TRUE)
plot(roc_improved_log, col = "green", lwd = 2, add = TRUE)
plot(roc_basic_log, col = "orange", lwd = 2, add = TRUE)
plot(roc_tree, col = "purple", lwd = 2, add = TRUE)

legend("bottomright", 
       legend = c("Random Forest", "Lasso", "Improved Log", "Basic Log", "Decision Tree"),
       col = c("red", "blue", "green", "orange", "purple"), 
       lwd = 2)

auc_scores <- c(
  Random_Forest = auc(roc_rf),
  Lasso = auc(roc_lasso),
  Improved_Log = auc(roc_improved_log),
  Basic_Log = auc(roc_basic_log),
  Decision_Tree = auc(roc_tree)
)

auc_table <- data.frame(Model = names(auc_scores), AUC = as.numeric(auc_scores))

auc_table <- auc_table[order(-auc_table$AUC), ]
print("Model Performance Ranking (by AUC):")
print(auc_table)

