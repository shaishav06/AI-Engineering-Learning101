"""
Topic 04 Interactive Student Micro-Exercises & Practice Challenges
Run this script using standard Python to test your implementations for Level 1 & Level 2 exercises!
No external package installation required!
"""

import math
import re

# =====================================================================
# 🟢 Micro-Exercise 1: Single Artificial Neuron Function
# =====================================================================
def perceptron_forward(inputs, weights, bias, activation='relu'):
    """
    Computes forward pass of a single artificial neuron.
    """
    # Calculate weighted sum z = sum(w_i * x_i) + b
    z = sum(w * x for w, x in zip(weights, inputs)) + bias
    
    # Apply activation function
    if activation == 'relu':
        a = max(0.0, z)
    elif activation == 'sigmoid':
        a = 1.0 / (1.0 + math.exp(-z))
    elif activation == 'linear':
        a = z
    else:
        raise ValueError(f"Unknown activation: {activation}")
        
    return z, a

# Test Micro-Exercise 1
inputs_ex1 = [2.0, -1.0, 3.0]
weights_ex1 = [0.5, 1.2, -0.4]
bias_ex1 = 0.5
z_val, a_val = perceptron_forward(inputs_ex1, weights_ex1, bias_ex1, activation='relu')
print(f"[Micro-Ex 1] Inputs={inputs_ex1} -> Weighted Sum z={z_val:.2f}, ReLU Output a={a_val:.2f}")


# =====================================================================
# 🟢 Micro-Exercise 2: Gradient Descent Weight Update
# =====================================================================
def gradient_descent_update(weight, learning_rate, gradient):
    """
    Performs 1 step of Gradient Descent update on a weight parameter.
    """
    new_weight = weight - (learning_rate * gradient)
    return new_weight

# Test Micro-Exercise 2
w_old = 2.5
lr = 0.1
grad = 0.8
w_new = gradient_descent_update(w_old, lr, grad)
print(f"[Micro-Ex 2] Old Weight={w_old} | Gradient={grad} -> New Weight={w_new:.2f}")


# =====================================================================
# 🟢 Micro-Exercise 3: HTML Tag Stripping Regex
# =====================================================================
def remove_html_tags(text):
    """
    Strips HTML markup tags from raw text using Regular Expressions.
    """
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

# Test Micro-Exercise 3
sample_html = "<p>Hello <b>AI Engineering</b> Students! Welcome to <i>Topic 04</i>.</p>"
cleaned_html = remove_html_tags(sample_html)
print(f"[Micro-Ex 3] Raw HTML: {sample_html}")
print(f"             Cleaned:  {cleaned_html}")


# =====================================================================
# 🟡 Micro-Exercise 4: TF-IDF Calculation Function
# =====================================================================
def compute_tfidf(tf_count, total_words_in_doc, df_count, total_docs):
    """
    Computes TF-IDF score for a given term in a document.
    """
    tf = tf_count / total_words_in_doc
    idf = math.log10(total_docs / df_count)
    tfidf = tf * idf
    return tf, idf, tfidf

# Test Micro-Exercise 4
tf_val, idf_val, tfidf_val = compute_tfidf(tf_count=3, total_words_in_doc=10, df_count=5, total_docs=100)
print(f"[Micro-Ex 4] TF={tf_val:.2f}, IDF={idf_val:.4f} -> TF-IDF Score={tfidf_val:.4f}")
