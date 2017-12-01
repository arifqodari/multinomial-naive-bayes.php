<?php

require("MultinomialNaiveBayes.php");

$features = array(
    array(2, 2, 1, 0, 0, 0, 0, 0, 0),
    array(0, 0, 1, 1, 1, 1, 0, 0, 0),
    array(0, 1, 1, 0, 0, 0, 1, 1, 0),
    array(0, 0, 1, 0, 2, 0, 0, 0, 1)
);
$target = array(1, 0, 1, 0); 

$test_features = array(
    array(1, 2, 1, 0, 0, 0, 0, 0, 0),
    array(0, 0, 1, 1, 2, 1, 0, 0, 0)
);

$naive_bayes = new MultinomialNaiveBayes(2);
$naive_bayes->fit($features, $target);

echo "<h3>Train Naive Bayes</h3>";
$params = $naive_bayes->get_params();
echo "Naive Bayes parameters<br/>";
echo "Class labels: " . print_r(range(0, $params["n_classes"] - 1), true) . "<br/>";
echo "Likelihood parameters: " . print_r($params["thetas"], true) . "<br/>";
echo "Priors: " . print_r($params["priors"], true) . "<br/>";

$train_accuracy = $naive_bayes->accuracy($features, $target);
echo "Train accuracy: " . $train_accuracy . "<br/>";

$prediction = $naive_bayes->predict($test_features);
echo "<h3>Test Naive Bayes</h3>";
echo "Prediction result: " . print_r($prediction, true) . "<br/>";

?>
