package com.codecicles.autodiff;

import java.util.Map;

import static java.util.Collections.emptyMap;

public class AutoDiff {
    public Result calculate(double aValue, double bValue) {
        Node a = node(aValue);
        Node b = node(bValue);

        Node x = add(mul(a, b), a);
        Node y = add(mul(a, b), x);
        Node z = mul(x, y);

        z.pushGradientToDependencies(1.0);

        return new Result(z, a, b);
    }

    static Node mul(Node a, Node b) {
        // derivative of "x * CONSTANT_PART" is "CONSTANT_PART"
        return node(a.value * b.value, Map.of(
                a, b.value,
                b, a.value
        ));
    }

    static Node add(Node a, Node b) {
        // derivative of "x + CONSTANT_PART" is "1.0"
        return node(a.value + b.value, Map.of(
                a, 1.0,
                b, 1.0
        ));
    }

    static Node square(Node a) {
        // derivative of "x^2" is "2 * x"
        return node(a.value * a.value, Map.of(
                a, 2 * a.value
        ));
    }

    static Node sub(Node a, Node b) {
        // Here, we let the auto diff to do its magic :)
        return add(a, negate(b));
    }

    static Node negate(Node a) {
        // derivative of "-x" is "-1"
        return node(-a.value, Map.of(
                a, -1.0
        ));
    }

    static Node node(double value) {
        return new Node(value, emptyMap());
    }

    static Node node(double value, Map<Node, Double> pdWrt) {
        return new Node(value, pdWrt);
    }

    static class Node {
        final double value;
        final Map<Node, Double> pdWrt;

        double gradient;

        public Node(double value, Map<Node, Double> pdWrt) {
            this.value = value;
            this.pdWrt = pdWrt;
        }

        @Override
        public String toString() {
            return "Node{" +
                    "value=" + value +
                    ", gradient=" + gradient +
                    '}';
        }

        void pushGradientToDependencies(double localGradientOfParent) {
            this.gradient += localGradientOfParent;

            for (Map.Entry<Node, Double> e : pdWrt.entrySet()) {
                Node dependency = e.getKey();
                Double pdWrtDependency = e.getValue();

                double localGradient = localGradientOfParent * pdWrtDependency;

                dependency.pushGradientToDependencies(localGradient);
            }
        }
    }

    static class Result {
        final Node z;
        final Node a;
        final Node b;

        public Result(Node z, Node a, Node b) {
            this.z = z;
            this.a = a;
            this.b = b;
        }

        @Override
        public String toString() {
            return "Result{" +
                    "z=" + z +
                    ", a=" + a +
                    ", b=" + b +
                    '}';
        }
    }
}
