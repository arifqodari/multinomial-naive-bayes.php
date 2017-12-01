<?php

class MultinomialNaiveBayes {

    private $n_classes;
    private $thetas;
    private $priors;

    public function __construct($n_classes, $thetas=NULL, $priors=NULL) {
        $this->n_classes = $n_classes;
        if (!is_null($thetas) && !is_null($priors)) {
            if ($n_classes != count($thetas) || $n_classes != count($priors))
                throw new Exception("Number of classes, thetas and prior dimension must be consistent!");
            $this->thetas = $thetas;
            $this->priors = $priors;
        }
    }

    public function get_params() {
        if ($this->is_trained())
            return array(
                "n_classes" => $this->n_classes,
                "thetas" => $this->thetas,
                "priors" => $this->priors);
        else
            return array(
                "n_classes" => $this->n_classes,
                "thetas" => NULL,
                "priors" => NULL);
    }

    public function fit($features, $target) {
        /*
         * Learning algorithm
         * features: NxD matrix
         * target: N-size vector
         */

        $params = array_map(
            function($class_label) use($features, $target) {
                // 1. find all documents with current class label
                $indices = array_keys($target, $class_label);

                // 2. calculate likelihood parameter / theta
                // with laplace smoothing
                $class_features = array_intersect_key($features, array_flip($indices));
                $m_cd = array_reduce(
                    array_map(NULL, ...$class_features),
                    function($result, $row) {
                        array_push($result, array_sum($row) + 1);
                        return $result;
                    },
                    array()
                );
                $m_c = array_sum($m_cd);

                $class_thetas = array_map(
                    function($element) use($m_c) {
                        return $element / $m_c;
                    },
                    $m_cd
                );

                // 3. calculate prior
                $class_prior = count($indices) / count($target);

                return array($m_cd, $class_prior);
            },
            range(0, $this->n_classes - 1)
        );

        $this->thetas = array_column($params, 0);
        $this->priors = array_column($params, 1);
    }

    public function predict($features) {
        /*
         * Prediction
         * Input features: NxD matrix
         * Output: N-size vector prediction result
         */

        if (!$this->is_trained()) return;

        $log_posteriors = array_map(
            function($class_label) use($features) {
                // calculate log posterior = log_likelihood + log prior
                $theta = $this->thetas[$class_label];
                $log_prior = log($this->priors[$class_label]);

                return array_map(
                    function($element) use($log_prior){
                        return array_sum($element) + $log_prior;
                    },
                    array_map(
                        function($element) use($theta) {
                            return array_map(
                                function($m, $n) {
                                    return $m * log($n);
                                },
                                $element,
                                $theta
                            );
                        },
                        $features
                    )
                );
            },
            range(0, $this->n_classes - 1)
        );

        // calculate prediction
        return array_map(
            function($class_probs) {
                return array_search(max($class_probs), $class_probs);
            },
            array_map(NULL, ...$log_posteriors)
        );
    }

    public function accuracy($features, $target) {
        /*
         * Calculate accuracy
         * features: NxD matrix
         * target: N-size vector
         */

        if (!$this->is_trained()) return;

        $prediction = $this->predict($features);

        return array_sum(array_map(
            function($p, $t) {
                return $p == $t ? 1 : 0;
            },
            $prediction,
            $target
        )) / count($target);
    }

    private function is_trained() {
        return isset($this->n_classes, $this->thetas, $this->priors);
    }

}

?>
