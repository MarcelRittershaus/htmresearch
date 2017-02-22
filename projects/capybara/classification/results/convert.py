""" Just a quick formatting script to convert HTM traces. """
import csv
import os

if not os.path.exists('traces'):
  os.makedirs('traces')

test_input = '../../htm/backup/trace_body_acc_x_inertial_signals_test.csv'
test_output = ('traces/test_body_acc_x_sp=True_tm=True_tp'
               '=False_SDRClassifier.csv')

train_input = '../../htm/backup/trace_body_acc_x_inertial_signals_train.csv'
train_output = ('traces/train_body_acc_x_sp=True_tm=True_tp'
                '=False_SDRClassifier.csv')

MAX_OUTPUT_ROWS = 40000



def format_csv(input, output, max_rows):
  with open(input, 'r') as fr:
    reader = csv.reader(fr)
    headers = reader.next()

    with open(output, 'w') as fw:
      writer = csv.writer(fw)
      writer.writerow(headers)
      counter = 0
      for row in reader:
        if counter < max_rows:
          writer.writerow(row)
          counter += 1
        else:
          return



format_csv(train_input, train_output, MAX_OUTPUT_ROWS)
format_csv(test_input, test_output, MAX_OUTPUT_ROWS)
