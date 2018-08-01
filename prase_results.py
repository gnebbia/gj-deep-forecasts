import tensorflow as tf

for e in tf.train.summary_iterator("/home/derek/deep-forecasts/logs/events.out.tfevents.1532561566.lambda-quad"):
    for v in e.summary.value:
        if v.tag == 'loss' or v.tag == 'accuracy':
            print(v.simple_value)
            raw_input('ok') 