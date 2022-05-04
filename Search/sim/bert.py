from bert import model_predict

####predict

seg_a = ""
seg_b = ""
prob, label = model_predict.single_predict(seg_a, seg_b)
