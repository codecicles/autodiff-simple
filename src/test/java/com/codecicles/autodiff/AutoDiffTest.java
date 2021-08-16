package com.codecicles.autodiff;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static com.codecicles.autodiff.AutoDiff.*;
import static org.junit.jupiter.api.Assertions.*;

class AutoDiffTest {
    @Test
    public void trainAgeProblem() {
        Random rnd = new Random(3);

        double adamValue = rnd.nextDouble();
        double belleValue = rnd.nextDouble();
        double lr = 0.01;

        for (int epoch = 0; epoch < 1000; epoch++) {
            Node adam = node(adamValue);
            Node belle = node(belleValue);
            Node totalSquareError = calcErrorOfAgeProblem(adam, belle);

            totalSquareError.pushGradientToDependencies(1.0);

            // How do we use the gradient?
            adamValue -= adam.gradient * lr;
            belleValue -= belle.gradient * lr;

            System.out.println(epoch + ":" +
                    " sqrErr=" + totalSquareError.value +
                    " adam=" + adamValue +
                    " belle=" + belleValue);
        }
    }

    private Node calcErrorOfAgeProblem(Node adam, Node belle) {
        // "Adam is 24 years older than Belle..."
        // Adam - 24 = Belle
        Node firstLeft = sub(adam, node(24));
        Node firstRight = belle;
        Node firstError = sub(firstLeft, firstRight);
        Node firstSquareError = square(firstError);

        // "...but in six years, Adam will be three times older than Belle"
        // (Adam + 6) = (Belle + 6) * 3
        Node secondLeft = add(adam, node(6));
        Node secondRight = mul(add(belle, node(6)), node(3));
        Node secondError = sub(secondLeft, secondRight);
        Node secondSquareError = square(secondError);

        Node totalSquaredError = add(firstSquareError, secondSquareError);

        return totalSquaredError;
    }

    @Test
    public void runCalculate() {
        AutoDiff autoDiff = new AutoDiff();

        double a = 5.0;
        double b = 3.0;
        Result result = autoDiff.calculate(a, b);

        Estimate gradientEstimate = calcNumericalEstimate(autoDiff, a, b);


        System.out.println("Result: " + result);
        System.out.println("GradientEstimate: " + gradientEstimate);
        assertEquals(gradientEstimate.gradientA, result.a.gradient, 0.0001);
        assertEquals(gradientEstimate.gradientB, result.b.gradient, 0.0001);
    }

    private Estimate calcNumericalEstimate(AutoDiff autoDiff, double a, double b) {
        double h = 0.0001;

        double gradientA;
        {
            double low = autoDiff.calculate(a - h/2.0, b).z.value;
            double high = autoDiff.calculate(a + h/2.0, b).z.value;
            gradientA = (high - low) / h;
        }

        double gradientB;
        {
            double low = autoDiff.calculate(a, b - h/2.0).z.value;
            double high = autoDiff.calculate(a, b + h/2.0).z.value;
            gradientB = (high - low) / h;
        }

        return new Estimate(gradientA, gradientB);
    }

    private class Estimate {
        private final double gradientA;
        private final double gradientB;

        public Estimate(double gradientA, double gradientB) {
            this.gradientA = gradientA;
            this.gradientB = gradientB;
        }

        @Override
        public String toString() {
            return "Estimate{" +
                    "gradientA=" + gradientA +
                    ", gradientB=" + gradientB +
                    '}';
        }
    }
}