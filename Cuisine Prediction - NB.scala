// Databricks notebook source
// DBTITLE 1,Model Imports
// Miscellaneous imports
import org.apache.spark.ml
import org.apache.spark.sql.functions._
import spark.implicits._

// Pipeline imports
import org.apache.spark.ml.{Pipeline, PipelineModel}
 
// Preprocessing imports
import org.apache.spark.ml.feature.{StringIndexer,RegexTokenizer,HashingTF,IDF,StandardScaler}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._

// Classification modeling imports
import org.apache.spark.ml.classification.{OneVsRest,LogisticRegression,LinearSVC,NaiveBayes}

// Model tuning imports
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

// Evaluation imports
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Imports for stratified train test split
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.DataFrameStatFunctions

// COMMAND ----------

// DBTITLE 1,Create a DataFrame from the imported csv
// Data imported through the UI
// All that needs to be done is take the imported data into a dataframe named 'df_recipes'
val df_recipes = spark.table("train_csv")

// Display transformed dataframe
df_recipes.show

// COMMAND ----------

// DBTITLE 1,Tokenization of Ingredients List
// To tokenize the lists of ingredients, we must use a special Regular Expression (Regex) pattern
// in order to separate the list into its different ingredients
// This is the process of tokenization

// Instantiate the RegexTokenizer class
val IngredientTokenizer = new RegexTokenizer()
    
   // Specify the field containing the strings to be tokenized, in this case the strings in "ingredients" column 
  .setInputCol("ingredients")
  
  // Specify an output column into which the results of our tokenization will be displayed
  .setOutputCol("ingredients_tokenized")

  // Specify a Regex pattern that will properly extract only the parts of the string with ingredient names
  // The strings in the ingredients column contain Python lists. They are mostly of the format ['ingredient1', 'ingredient2', 'ingredient3']
  // Wherever an ingredient contains an apostrophe, the ingredient will appear in the list as "ingredient4"; these will be properly tokenized
  // by the pattern specified below
  .setPattern("(?<=([\"']\\b))(?:(?=(\\\\?))\\2.)*?(?=\\1)").setGaps(false) 

// Apply our specified tokenization pattern to the original dataframe
// Storing the output into a new dataframe, 'df_recipes_tokenized'
val df_recipes_tokenized = IngredientTokenizer.transform(df_recipes)

// Display transformed dataframe
df_recipes_tokenized.show

// COMMAND ----------

// DBTITLE 1,Display Number of Occurrences for Top Ingredients
// The explode method will be used to break out tokenized ingredient lists into
// individual ingredients, where they will be stored in a new column
val columns = df_recipes_tokenized.columns.map(col) :+
  (explode(col("ingredients_tokenized")) as "ingredients_individual")

// Create dataframe featuring the new "ingredients_individual column"
val df_recipes_exploded = df_recipes_tokenized.select(columns: _*)

// Determine the frequency with which each ingredient is used in recipes using the "groupBy" function
// on the "ingredients_individual" column
val ingredients_count = df_recipes_exploded.groupBy("ingredients_individual").count().orderBy(desc("count"))

// Display the top 20 ingredients according to overall frequency
display(ingredients_count.limit(20))

// Observations: The presence of salt exceeds that of many other ingredients by alot.
// It is important to note that for these frequent ingredients, it is possible that
// they can be found very often in certain cuisines, while they can be quite absent
// in others

// COMMAND ----------

// DBTITLE 1,Display Percentage Count for Top Ingredients
// Create a new dataframe featuring a column listing the percent of time ingredients are found in recipes
val ingredients_frac_count = ingredients_count.withColumn("percent_count",( $"count" / df_recipes_exploded.count ) * 100)

// Display the top 20 ingredients according to the percentage described above
display(ingredients_frac_count.limit(20))

// Observations: Similar to previous code cell

// COMMAND ----------

// DBTITLE 1,Display Number of Occurrences for Cuisine Types
// Create a new dataframe featuring a column listing the frequencies of different cuisines
val cuisine_count = df_recipes_tokenized.groupBy("cuisine").count().orderBy(desc("count"))

// Display all cuisines with their corresponding frequencies
display(cuisine_count)

// Observations: We have an imbalanced dataset, and to produce better model results, we
// will need to process our data for training accordingly

// COMMAND ----------

// DBTITLE 1,Display Percentage Count for Cuisine Types
// Another dataframe created with the "label_percentage_over_total" column added
val cuisine_frac_count = cuisine_count.withColumn("percent_count",( $"count" / df_recipes_tokenized.count ) * 100)

// Display all cuisines with their corresponding appearance percentages
display(cuisine_frac_count)

// Observations: Similar to previous code cell

// COMMAND ----------

// DBTITLE 1,Number of Unique Ingredients
// We will apply the explode method to determine the amount of unique ingredients
// in the dataset

val columns = df_recipes_tokenized.columns.map(col) :+
  (explode(col("ingredients_tokenized")) as "ingredients_individual")

val df_recipes_exploded = df_recipes_tokenized.select(columns: _*)

val ingredients_unique = df_recipes_exploded.select("ingredients_individual").distinct.count.toInt

// COMMAND ----------

// DBTITLE 1,Convert Target Values into Integers
// To begin the process of converting string values within the dataframe into numerical values
// onto which machine learning models can be applied, we will start by converting the target values,
// (the cuisine which we must predict from the recipes), we will use the StringIndexer class

// Instantiate the StringIndexer class
val target_value_indexer = new StringIndexer()
  
  // Input column are the cuisines which correspond to set of recipes
  .setInputCol("cuisine")

  // Output column of IDs corresponding to each cuisine string in the dataset
  .setOutputCol("label")

// Apply our specified tokenization pattern to the original dataframe using the fit and transform methods,
// storing the output into a new dataframe, 'df_recipes_indexed'
val df_recipes_indexed_fit = target_value_indexer.fit(df_recipes_tokenized)

val df_recipes_indexed = df_recipes_indexed_fit.transform(df_recipes_tokenized)

// Display transformed dataframe
df_recipes_indexed.show

// COMMAND ----------

// DBTITLE 1,Applying Custom Weights to Cuisines
// Create a partition window to be used for determining the count of each cuisine type
//val partitionWindow_cuisine = Window.partitionBy($"cuisine")

// Determine counts for each wine variety using the partition window
//val cuisine_count = count("cuisine").over(partitionWindow_cuisine)

// New dataframe with a "label_count column added"
//val df_recipes_weighted_count = df_recipes_indexed.withColumn("cuisine_count", cuisine_count)

// Add column with total number of different recipes
//val df_recipes_weighted_count_2 = df_recipes_weighted_count.withColumn("total_cuisines", lit(df_recipes_weighted_count.distinct.count))

// Add column with weights of recipes
//val df_recipes_weighted_1 = df_recipes_weighted_count_2.withColumn("weight_column",$"total_samples"/($"cuisine_count" * 10))

// Store only necessary columns for remaining stages into a new dataframe
//val df_recipes_weighted = df_recipes_weighted_1.select("_c0","id","cuisine","sorted_ingredients_list","label","weight_column")

// COMMAND ----------

// DBTITLE 1,Vectorization and Encoding of Features (1. Term Frequency of Ingredients)
// In the first step of converting our tokenized ingredients into vectors,
// we will assign a term frequencies to each ingredient found in each recipe
// Note that since ingredients are likely not to be used twice within the same recipe, 
// the term frequencies should all be 1. Nevertheless we will proceed with the usual
// TF-IDF encoding process

// Instantiate the HashingTF class
val ingredients_HashingTF = new HashingTF()

  // The lists of tokenized ingredients
  .setInputCol("sorted_ingredients_list")

  // The column of sparse vectors containing term frequencies
  .setOutputCol("ingredients_tf")

  // Number of unique ingredients present in the model
  .setNumFeatures(ingredients_unique)
  //.setNumFeatures(150)

// COMMAND ----------

// DBTITLE 1,Vectorization and Encoding of Features (2. Inverse Document Frequency of Ingredients)
// Instantiate the IDF class
val ingredients_IDF = new IDF()

  // Input the term frequencies of each ingredient in the recipes
  .setInputCol(ingredients_HashingTF.getOutputCol)
  
  // Output the term frequencies scaled by the IDF factor into the column "ingredients_tfidf"
  .setOutputCol("features")

// COMMAND ----------

// DBTITLE 1,Apply Standard Scaler to the Data
// Scale the data to remove the mean and have unit variance
val scaler = new StandardScaler()
  .setInputCol(ingredients_IDF.getOutputCol)
  .setOutputCol("features")

// COMMAND ----------

// DBTITLE 1,Stratified Train-test Split
// This function will produce a "stratified" train-test split, which can be useful when dealing with 
// an imbalanced dataset such as this one
def train_test_split_stratified(test_fraction: Double, input_dataframe: DataFrame, stratified: Boolean): (DataFrame,DataFrame) = {

  if(stratified == true){
    
    // Extract list of column names from input dataframe
    val column_names = input_dataframe.columns.toSeq
    
    // Extract list of cuisines from the dataset
    val cuisine_array = input_dataframe.select("cuisine").distinct.orderBy(asc("cuisine")).collect().map(array_element => array_element(0))
    val cuisine_list = cuisine_array.toList
    
    // Returns a list of the required fraction for training data for each of the classes present in the dataset
    // In the case of a stratified split, this fraction should be equal among all the classes
    val training_fraction_list = List.fill(cuisine_list.length)(1 - test_fraction)

    // Takes the provided list of cuisines and maps each of them to the required training fraction
    // This mapping is required as an input of the sampleBy method
    val training_factors = (cuisine_list zip training_fraction_list).toMap

    // Perform a stratified split on the original dataset and store the training set
    val training_set = input_dataframe.stat.sampleBy("cuisine", training_factors, 42)

    // Now, the rows from the original dataset which are not found in the training set
    // will be allocated to the test set
    val training_set_join =  training_set.select(training_set.columns.map { c => training_set.col(c).as( c + "_1") } : _* )

    // Perform an outer join between the original dataframe and the newly created dataset
    // All rows excluding those of the training set will be set to those of the test sets in the two
    // commands below
    val df_training_match = input_dataframe.join(training_set_join, input_dataframe.col("id") === training_set_join.col("id_1"),"outer")

    val test_set = df_training_match.filter("id_1 is null").select(column_names.map(c => col(c)): _*)
    // Return training and test sets
    return (training_set,test_set)
    
  }
  
  else {
     // Return training and test sets
    val Array(training_set_random, test_set_random) = input_dataframe.randomSplit(Array(1 - test_fraction, test_fraction), 42)
    return (training_set_random,test_set_random)  
    
  }
  

}

// Split the training and test data (80% training, 20% testing) by calling the above function
val (recipes_training, recipes_test) = train_test_split_stratified(0.2,df_recipes_indexed,true)

// COMMAND ----------

// DBTITLE 1,Confirmation of Absence of Data-leakage
// If the generated dataframe is empty, it is confirmed that no data leakage occurred during the train-test split
val train_test_match = recipes_training.join(recipes_test, recipes_training.col("_c0") === recipes_test.col("_c0")).show

// COMMAND ----------

// DBTITLE 1,Searching for Duplicate Recipes within the Same Cuisine
// Sort each recipe's ingredient list in alphabetical order
val df_ingredients_sorted = recipes_training.withColumn("sorted_ingredients_list", sort_array($"ingredients_tokenized"))

// Determine duplicate ingredient lists within the same cuisine
val duplicates_same_cuisine = df_ingredients_sorted.groupBy("sorted_ingredients_list", "cuisine").count().where($"count" > 1)

// Display results
duplicates_same_cuisine.count

// COMMAND ----------

// DBTITLE 1,Search for all Duplicate Recipes
// Duplicate ingredient lists with the same or different cuisine 
val duplicates_general = df_ingredients_sorted.groupBy("sorted_ingredients_list").count().where($"count" > 1)   

// Display results
duplicates_general.count
// There are more rows here in than the cell above: this means there are duplicate ingredient lists with DIFFERENT cuisines
// The numbers we've returned in this code cell and the previous one imply that there are 12 recipes which are duplicates but belong to 
// different cuisines.

// COMMAND ----------

// DBTITLE 1,Remove Duplicates With Different Cuisines (Type 1)
// Manually remove duplicate ingredient lists with DIFFERENT cuisine assigned to it (only 12 pairs --> 24 rows)
val df_no_duplicate_1 = df_ingredients_sorted.where($"sorted_ingredients_list"=!=Array("all-purpose flour", "butter", "milk", "parmesan cheese", "pepper", "salt") &&
$"sorted_ingredients_list"=!=Array("brown sugar","carrots","garlic chili sauce", "green onions", "medium shrimp uncook", "napa cabbage", "rice noodles", "sesame oil", "soy sauce") &&
$"sorted_ingredients_list"=!=Array("andouille sausage","applewood smoked bacon","beef broth","boneless chicken skinless thigh","cayenne pepper", "celery ribs", "chili powder", "chopped fresh thyme", "diced tomatoes", "flat leaf parsley", "green bell pepper", "green onions","long grain white rice","onions","paprika","red bell pepper","sausages", "tasso") && $"sorted_ingredients_list"=!=Array("active dry yeast", "all purpose unbleached flour", "salt", "sugar", "warm water") &&           
$"sorted_ingredients_list"=!=Array("bay leaves", "boneless pork shoulder", "dried oregano", "fresh lime juice", "ground black pepper", "ground cumin", "kosher salt", "onions", "orange", "water") &&
$"sorted_ingredients_list"=!=Array("cinnamon sticks", "corn starch", "large egg yolks", "lemon rind", "salt", "sugar", "whole milk") &&
$"sorted_ingredients_list"=!=Array("kosher salt", "lemon", "water") &&
$"sorted_ingredients_list"=!=Array("chili oil", "rice vinegar", "soy sauce") && 
$"sorted_ingredients_list"=!=Array("active dry yeast", "all-purpose flour", "salt", "vegetable oil", "warm water", "white sugar") &&  
$"sorted_ingredients_list"=!=Array("butter") &&   
$"sorted_ingredients_list"=!=Array("all-purpose flour", "almond extract", "baking powder", "butter", "powdered sugar", "salt", "vanilla extract") &&   
$"sorted_ingredients_list"=!=Array("all-purpose flour", "baking powder", "butter", "powdered sugar", "salt", "toasted pecans", "vanilla extract"))

df_no_duplicate_1.count

// COMMAND ----------

// DBTITLE 1,Remove a Duplicate With Same Cuisines (Type 2)
// Now, remove duplicate ingredient lists with SAME cuisine assigned to it (just remove ONE of the two)
val recipes_training_no_duplicate = df_no_duplicate_1.dropDuplicates("sorted_ingredients_list")
recipes_training_no_duplicate.count()

// About 500 duplicates gone!

// COMMAND ----------

// DBTITLE 1,Specify Model Type and Hyperparameters (Option 1 - Logistic Regression)
// Instantiate Logistic Regression model
val lr = new LogisticRegression()
  .setRegParam(0.01)
  .setMaxIter(50)

// COMMAND ----------

// DBTITLE 1,Specify Model Type and Hyperparameters (Option 2 - Linear SVC)
// Instantiate LSVC model. LVSC will be used in conjunction with One vs Rest training
val lsvc = new LinearSVC()

// Instantiate One vs. Rest Training Methodology
val ovr = new OneVsRest()
  ovr.setClassifier(lsvc)

// COMMAND ----------

// DBTITLE 1,Specify Model Type and Hyperparameters (Option 3 - Naive Bayes)
// Instantiate Naive Bayes model. Naive Bayes supports Multinomial Classification
val nb = new NaiveBayes() 
  .setSmoothing(2)
  //.setWeightCol("weight_column")


// COMMAND ----------

// DBTITLE 1,Build Pipeline
// Construct a pipeline consisting of determining term frequency counts (HashingTF)
// multiple frequencies by IDF factors (ingredients_IDF), and then apply One vs. Rest/One vs. All
// Classification using Linear SVC
val pipeline = new Pipeline()
  .setStages(Array(ingredients_HashingTF,ingredients_IDF,nb))

// COMMAND ----------

// DBTITLE 1,Build Hyperparameter Grid for Tuning
// Build a grid of hyperparameter combinations to iterate over in order to find the most accurate model
// when evaluating the training set's ability to predict validation data targets 
val paramGrid = new ParamGridBuilder()
  //.addGrid(lsvc.regParam, Array[Double](0.01))
  //.addGrid(lsvc.maxIter, Array(50))
  .addGrid(ingredients_HashingTF.numFeatures,Array(100,500,1000,3000,ingredients_unique))
  .addGrid(nb.smoothing,Array[Double](1,2,3,4,5,6,7))
  .build()

// Output the paramGrid
paramGrid

// COMMAND ----------

// DBTITLE 1,Cross Validation on the Training Data
// In this case the estimator is simply the linear regression
// Cross Validation requires an Estimator, a set of Estimator ParamMaps (hyperparameter grid to iterate through)
// and an Evaluator.

// Set the training evaluator
val training_evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("weightedFMeasure")

// Instantiate CrossValidator class
val cv = new CrossValidator()

  .setEstimator(pipeline)
  .setEvaluator(training_evaluator)
  .setEstimatorParamMaps(paramGrid)

  .setNumFolds(5)
//  .setParallelism(3)
 
// Run cross validation, and choose the best set of parameters.
val cv_model = cv.fit(recipes_training_no_duplicate)

// COMMAND ----------

// DBTITLE 1,Cross Validation Evaluation
// Returns the average evaluations produced from all parameter combinations possible from the ParamGrid
cv_model.avgMetrics

// COMMAND ----------

// DBTITLE 1,Optimal Model Parameters
// Return the optimal parameters used in model development
//println("Optimal hyperparameters from gridsearch:")
//println("NB Smoothing: " + cv_model.bestModel.get(nb.smoothing).getOrElse(0))
//println("Number of features: " + cv_model.bestModel.get(ingredients_HashingTF.numFeatures).getOrElse(0))


//val nb_smoothing_optimal = cv_model.bestModel.asInstanceOf[PipelineModel].stages(0).get(ingredients_HashingTF.numFeatures).getOrElse(0)
//val num_features_optimal = cv_model.bestModel.asInstanceOf[PipelineModel].stages(2).get(nb.smoothing).getOrElse(0)

// COMMAND ----------

// DBTITLE 1,Display Prediction vs. Labels
// Output class predictions for comparison against labels

val recipes_test_input = recipes_test.withColumnRenamed("ingredients_tokenized","sorted_ingredients_list")

val prediction = cv_model.transform(recipes_test_input)


// COMMAND ----------

// DBTITLE 1,Evaluating Model's Performance Against Test Data (Accuracy)
// Declare the evaluation metric to be used
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

// Evaluate the prediction performance
val accuracy = evaluator.evaluate(prediction)

// Display results
println("Accuracy = " + accuracy * 100 + "%")

// COMMAND ----------

// DBTITLE 1,Define Metrics Objects to Evaluate the Model
//Define a predictionAndLabels object
val predictionAndLabels = prediction.select($"prediction", $"label").as[(Double, Double)].rdd
 
// Instantiate metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// COMMAND ----------

// DBTITLE 1,Metrics Set 1
// Display weighted precision, weighted recall, weighted F1 score, and weighted false positive rate
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")

// COMMAND ----------

// DBTITLE 1,Macro-Averaged F1 Score
// Function to calculate macro-averaged F1 scores
def F1_macro_average(metrics_input: org.apache.spark.mllib.evaluation.MulticlassMetrics): Double = {
  // From multiclass metrics object extract labels into a list
  val labels = metrics_input.labels
  // Initialize F1_sum value as 0
  var F1_sum: Double = 0
  // Add each label's F1 score together
  labels.foreach { l =>
    F1_sum = metrics_input.fMeasure(l) + F1_sum
    
  }
  // Return the macro-averaged F1 (sum of F1-scores divided by number of classes)
  return(F1_sum/metrics_input.labels.length)
}

// Test output
val F1_macro_avg = F1_macro_average(metrics)

// COMMAND ----------

// DBTITLE 1,Confusion Matrix
// Confusion matrix
val cols = (0 until metrics.confusionMatrix.numCols).toSeq
val cm = metrics.confusionMatrix.transpose.colIter.toSeq.map(_.toArray).toDF("arr")
val cm2 = cols.foldLeft(cm)((cm, i) => cm.withColumn("_" + (i+1), $"arr"(i))).drop("arr")
display(cm2)

// COMMAND ----------

// DBTITLE 1,Alternative: Skip Cross Validation
// Fit the pipeline model to the training data
val pipeline_model = pipeline.fit(recipes_training_no_duplicate)

// COMMAND ----------

// DBTITLE 1,Training and Validation Macro F1-Score vs. Naive Bayes Smoothing
// The following code will be used to output data that will be represented in a plot, for the purpose
// of compare training and validation macro F1-scores with increasing smoothing factor
for( n <- 1 to 7){
  val nb_loop = new NaiveBayes() 
    .setSmoothing(n)

  val pipeline_loop = new Pipeline()
    .setStages(Array(ingredients_HashingTF,ingredients_IDF,nb_loop))

  val pipeline_loop_model = pipeline_loop.fit(recipes_training_no_duplicate)
  // Predict cuisines for recipes of the test data
  val train_prediction_no_cv = pipeline_loop_model.transform(recipes_training_no_duplicate).select("features", "label", "prediction")
  
  val test_prediction_no_cv = pipeline_loop_model.transform(recipes_test_input).select("features", "label", "prediction")
  
  // Define a predictionAndLabels object
  val train_predictionAndLabels_no_cv = train_prediction_no_cv.select($"prediction", $"label").as[(Double, Double)].rdd
  val test_predictionAndLabels_no_cv = test_prediction_no_cv.select($"prediction", $"label").as[(Double, Double)].rdd
  
  // Instantiate metrics object
  val train_metrics_no_cv = new MulticlassMetrics(train_predictionAndLabels_no_cv)
  val test_metrics_no_cv = new MulticlassMetrics(test_predictionAndLabels_no_cv)
  val F1_macro_avg_train = F1_macro_average(train_metrics_no_cv)
  val F1_macro_avg_test = F1_macro_average(test_metrics_no_cv) 
  
  println("F1-macro average (training) n = ",n,": ",F1_macro_avg_train)
  println("F1-macro average (testing) n = ",n,": ",F1_macro_avg_test)
}

// COMMAND ----------

// DBTITLE 1,Display Predictions vs. Targets
// Predict cuisines for recipes of the test data
val prediction_no_cv = pipeline_model.transform(recipes_test_input).select("features", "label", "prediction")

// Display results
prediction_no_cv.show

// COMMAND ----------

//Define a predictionAndLabels object
val predictionAndLabels_no_cv = prediction_no_cv.select($"prediction", $"label").as[(Double, Double)].rdd
 
// Instantiate metrics object
val metrics_no_cv = new MulticlassMetrics(predictionAndLabels_no_cv)

// COMMAND ----------

// Display weighted precision, weighted recall, weighted F1 score, and weighted false positive rate
println(s"Weighted precision: ${metrics_no_cv.weightedPrecision}")
println(s"Weighted recall: ${metrics_no_cv.weightedRecall}")
println(s"Weighted F1 score: ${metrics_no_cv.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics_no_cv.weightedFalsePositiveRate}")

// COMMAND ----------

// Confusion matrix
val cols_no_cv = (0 until metrics_no_cv.confusionMatrix.numCols).toSeq
val cm_no_cv = metrics_no_cv.confusionMatrix.transpose.colIter.toSeq.map(_.toArray).toDF("arr")
val cm2_no_cv = cols.foldLeft(cm_no_cv)((cm_no_cv, i) => cm_no_cv.withColumn("_" + (i+1), $"arr"(i))).drop("arr")
display(cm2)
