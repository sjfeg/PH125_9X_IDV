library(dslabs)
library(tidyverse)
library(caret)
library(plyr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(kableExtra)

# ----------------------------------------------------------------------------------------
# ---------------------------- Importing and Tidying Data --------------------------------
# ----------------------------------------------------------------------------------------

# The csv datafile (mushrooms.csv) is assumed to be in the same directory than this code
# The zip file can be downloaded at https://www.kaggle.com/uciml/mushroom-classification/downloads/mushroom-classification.zip
# I could not find an easy way to download directly kaggle datasets from the code
# I found plenty of articles relating to this issue on Google, but the solutions go above my paygrade

raw <- read.csv("./mushrooms.csv")

# Expliciting factor levels
raw$class <- mapvalues(raw$class, from = c("e", "p"), 
                       to = c("edible", "poisonous"))
raw$cap.shape <- mapvalues(raw$cap.shape, from = c("b", "c", "x", "f", "k", "s"), 
                           to = c("bell", "conical", "convex", "flat", "knobbed", "sunken"))
raw$cap.surface <- mapvalues(raw$cap.surface, from = c("f", "g", "y", "s"), 
                             to = c("fibrous", "grooves", "scaly", "smooth"))
raw$cap.color <- mapvalues(raw$cap.color, from = c("n", "b", "c", "g", "r", "p", "u", "e", "w", "y"), 
                           to = c("brown", "buff", "cinnamon", "grey", "green", "pink", "purple", "red", "white", "yellow"))
raw$bruises <- mapvalues(raw$bruises, from = c("t", "f"), 
                         to = c("bruises", "no"))
raw$odor <- mapvalues(raw$odor, from = c("a", "l", "c", "y", "f", "m", "n", "p", "s"), 
                      to = c("almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"))
raw$gill.attachment <- mapvalues(raw$gill.attachment, from = c("a", "d", "f", "n"), 
                                 to = c("attached", "descending", "free", "notched"))
raw$gill.spacing <- mapvalues(raw$gill.spacing, from = c("c", "w", "d"), 
                              to = c("close", "crowded", "distant"))
raw$gill.size <- mapvalues(raw$gill.size, from = c("b", "n"), 
                           to = c("broad", "narrow"))
raw$gill.color <- mapvalues(raw$gill.color, from = c("k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"), 
                           to = c("black", "brown", "buff", "chocolate", "grey", "green", "orange", "pink", "purple", "red", "white", "yellow"))
raw$stalk.shape <- mapvalues(raw$stalk.shape, from = c("e", "t"), 
                             to = c("enlarging", "tapering"))
raw$stalk.root <- mapvalues(raw$stalk.root, from = c("b", "c", "u", "e", "z", "r", "?"), 
                            to = c("bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"))
raw$stalk.surface.above.ring <- mapvalues(raw$stalk.surface.above.ring, from = c("f", "y", "k", "s"), 
                                          to = c("fibrous", "scaly", "silky", "smooth"))
raw$stalk.surface.below.ring <- mapvalues(raw$stalk.surface.below.ring, from = c("f", "y", "k", "s"), 
                                          to = c("fibrous", "scaly", "silky", "smooth"))
raw$stalk.color.above.ring <- mapvalues(raw$stalk.color.above.ring, from = c("n", "b", "c", "g", "o", "p", "e", "w", "y"), 
                           to = c("brown", "buff", "cinnamon", "grey", "orange", "pink", "red", "white", "yellow"))
raw$stalk.color.below.ring <- mapvalues(raw$stalk.color.below.ring, from = c("n", "b", "c", "g", "o", "p", "e", "w", "y"), 
                                        to = c("brown", "buff", "cinnamon", "grey", "orange", "pink", "red", "white", "yellow"))
raw$veil.type <- mapvalues(raw$veil.type, from = c("p", "u"), 
                           to = c("partial", "universal"))
raw$veil.color <- mapvalues(raw$veil.color, from = c("b", "o", "w", "y"), 
                            to = c("brown", "orange", "white", "yellow"))
raw$ring.number <- mapvalues(raw$ring.number, from = c("n", "o", "t"), 
                             to = c("none", "one", "two"))
raw$ring.type <- mapvalues(raw$ring.type, from = c("c", "e", "f", "l", "n", "p", "s", "z"), 
                      to = c("cobwebby", "evanescent", "flaring", "large", "none", "pendant", "sheating", "zone"))
raw$spore.print.color <- mapvalues(raw$spore.print.color, from = c("k", "n", "b", "h", "r", "o", "u", "w", "y"), 
                            to = c("black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"))
raw$population <- mapvalues(raw$population, from = c("a", "c", "n", "s", "v", "y"),
                            to = c("abundant", "clustered", "numerous", "scattered", "several", "solitary"))
raw$habitat <- mapvalues(raw$habitat, from = c("g", "l", "m", "p", "u", "w", "d"), 
                            to = c("grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"))

# Removing veil.type (always the same value)
raw <- select(raw, -veil.type)
mushrooms <- raw
rm(raw)

# Creating Train and Test set
set.seed(1)
index <- createDataPartition(mushrooms$class, times = 1, p = 0.4, list = FALSE)
test_set <- mushrooms[index, ]
train_set <- mushrooms[-index, ]
y <- test_set$class

# ----------------------------------------------------------------------------------------
# -------------------------------- LOGISTIC REGRESSION -----------------------------------
# ----------------------------------------------------------------------------------------
glm_train <- glm(class ~ ., data = train_set, family = "binomial", maxit = 100)
y_hat_glm <- predict(glm_train, newdata = test_set, type = "response")
y_hat_glm <- as.factor(ifelse(y_hat_glm > 0.5, "poisonous", "edible"))
table(y, y_hat_glm)

# Storing results
cfm <- confusionMatrix(y_hat_glm, y)
results <- tibble(method = "Logistic regression", 
                  Specificity = cfm$byClass[["Specificity"]], 
                  Sensitivity = cfm$byClass[["Sensitivity"]], 
                  Accuracy = cfm$overall[["Accuracy"]])


# ----------------------------------------------------------------------------------------
# -------------------------------- K NEAREST NEIGHBORS -----------------------------------
# ----------------------------------------------------------------------------------------
knn_train <- train(class ~ ., method = "knn", 
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3, 11, 2)))
ggplot(knn_train, highlight = TRUE) + scale_x_continuous(breaks = c(3,7,11))
y_hat_knn <- predict(knn_train, newdata = test_set, type = "raw")
table(y, y_hat_knn)

# Storing results
cfm <- confusionMatrix(y_hat_knn, y)
results <- bind_rows(results, tibble(method = "k-nearest neighbors", 
                  Specificity = cfm$byClass[["Specificity"]], 
                  Sensitivity = cfm$byClass[["Sensitivity"]], 
                  Accuracy = cfm$overall[["Accuracy"]]))


# ----------------------------------------------------------------------------------------
# ---------------------------------  DECISION TREE  --------------------------------------
# ----------------------------------------------------------------------------------------
rpart_train <- rpart(class ~ ., data = train_set, method = "class")
y_hat_rpart <- predict(rpart_train, newdata = test_set, type="class")
table(y, y_hat_rpart)

# Storing results
cfm <- confusionMatrix(y_hat_rpart, y)
results <- bind_rows(results, tibble(method = "Decision tree", 
                                     Specificity = cfm$byClass[["Specificity"]], 
                                     Sensitivity = cfm$byClass[["Sensitivity"]], 
                                     Accuracy = cfm$overall[["Accuracy"]]))

# Adding a penalty matrix to minimize false negative
penalty.matrix <- matrix(c(0,10,1,0), nrow = 2, ncol = 2)
rpart2_train <- rpart(class ~ ., data = train_set, method = "class", parms = list(loss = penalty.matrix))
y_hat_rpart2 <- predict(rpart2_train, newdata = test_set, type="class")
table(y, y_hat_rpart2)

# Plotting the tree
rpart.plot(rpart2_train, 
           type = 5,               # var names within nodes
           tweak = 0.8,            # reduce text size
           under = TRUE,           # puts extra data under the box
           extra = 101,            # show percentages and number of observations
           box.palette = "GnBu",   # color scheme
           branch.lty = 3)         # dotted branch lines

# Storing results
cfm <- confusionMatrix(y_hat_rpart2, y)
results <- bind_rows(results, tibble(method = "Decision tree with penalty", 
                                     Specificity = cfm$byClass[["Specificity"]], 
                                     Sensitivity = cfm$byClass[["Sensitivity"]], 
                                     Accuracy = cfm$overall[["Accuracy"]]))

# Check accuracy on the training set
check <- predict(rpart2_train, newdata = train_set, type = "class")
table(train_set$class, check)


# ----------------------------------------------------------------------------------------
# ---------------------- LOGISTIC REGRESSION WITH 5 PREDICTORS ---------------------------
# ----------------------------------------------------------------------------------------

form <- as.formula("class ~ odor + spore.print.color + population + gill.size + habitat")
glm5P_train <- glm(form, data = train_set, family = "binomial", maxit = 100)
y_hat_glm5P <- predict(glm5P_train, newdata = test_set, type = "response")
y_hat_glm5P <- as.factor(ifelse(y_hat_glm5P > 0.5, "poisonous", "edible"))
table(y, y_hat_glm5P)

# Storing results
cfm <- confusionMatrix(y_hat_glm5P, y)
results <- bind_rows(results, tibble(method = "5 predictors logistic regression", 
                                     Specificity = cfm$byClass[["Specificity"]], 
                                     Sensitivity = cfm$byClass[["Sensitivity"]], 
                                     Accuracy = cfm$overall[["Accuracy"]]))

# ----------------------------------------------------------------------------------------
# ------------------------------------ RESULTS -------------------------------------------
# ----------------------------------------------------------------------------------------
results
