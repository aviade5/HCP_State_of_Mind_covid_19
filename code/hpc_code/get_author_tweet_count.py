import sqlite3 as sql
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

database_path = "/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db"

conn = sql.connect(database_path)
cur = conn.cursor()
query = "SELECT author, COUNT(*) FROM posts GROUP BY author"
results = cur.execute(query).fetchall()

tweet_count_list = [tweet_count for (_, tweet_count) in results]

# how many users with most posts have 10% of the tweets?

total_number_of_posts = 16616970

num_users = 0
tweet_sum = 0

total_tweet_percentage = 0.1

for c in sorted(tweet_count_list, reverse=True):
  print(c)
  tweet_sum += c
  num_users += 1
  if tweet_sum >= total_tweet_percentage * total_number_of_posts:
    break

print("{} users with the most posts account for {}% of all tweets".format(num_users, tweet_sum / total_number_of_posts * 100.0))

count_count_dict = {count : tweet_count_list.count(count) for count in set(tweet_count_list)}

mpl.rcParams['font.size'] = 14

plt.subplots(figsize=(10, 6))
#plt.title('Logarithm of author count per number of posts')
plt.xlabel('#posts')
plt.ylabel('#HCPs')
plt.yscale("log")
plt.xscale("log")
plt.scatter([num_tweets for (num_tweets, _) in count_count_dict.items()], [count for (_, count) in count_count_dict.items()])
plt.savefig('log_author_count_per_log_number_of_posts.png')