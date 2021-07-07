import pandas as pd

dfs = [pd.read_csv('./tweets_to_classify/job{}_tweets.csv'.format(i)) for i in range(1, 7)]
length_sum = sum([len(df) for df in dfs])
print('Total length of all dfs:', length_sum, flush=True)

post_id_set = set()

for df in dfs:
  post_id_set |= set(df['post_id'])

print('Total length of union of tweet ids:', len(post_id_set), flush=True)

if length_sum == len(post_id_set):
  print('Looks good!', flush=True)
else:
  print('We\'ve got trouble :(', flush=True)