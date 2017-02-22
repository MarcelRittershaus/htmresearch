""" Just a quick formatting script """
import csv

test_input = 'traces/trace_body_acc_x_inertial_signals_test.csv'
test_output = ('traces/test_body_acc_x_sp=True_tm=True_tp'
               '=False_SDRClassifier.csv')

train_input = 'traces/trace_body_acc_x_inertial_signals_train.csv'
train_output = ('traces/train_body_acc_x_sp=True_tm=True_tp'
                '=False_SDRClassifier.csv')

max_rows = 40000



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



format_csv(train_input, train_output, max_rows)
format_csv(test_input, test_output, max_rows)
