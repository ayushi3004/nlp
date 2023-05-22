import matplotlib.pyplot as plt

x = [1000, 5000, 10000, 20000, 30000, 40000]
w_err_y = [0.28165117032679415, 0.10825834434279732,  0.08143679736769947,  0.06558316923000225, 0.05718274048408405, 0.05409178153899843]
sent_err_y = [0.9482352941176471,  0.8, 0.7494117647058823,  0.7035294117647058,  0.6676470588235294, 0.6558823529411765]

plt.plot(x, w_err_y, label = "Word error rate")
plt.plot(x, sent_err_y, label = "Sentence error rate")

plt.xlabel('Training Dataset Size')
plt.ylabel('Error Rate')
plt.legend()
plt.show()