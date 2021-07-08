import sqlite3 as sql
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

database_path = "/gpfs0/rami/users/iliapl/data/databases/53k_individual_hcps_70_percent_confidence_tweets.db"

conn = sql.connect(database_path)
cur = conn.cursor()
query = "SELECT author, COUNT(*) FROM posts WHERE date > date('2019-12-31') GROUP BY author"
results = cur.execute(query).fetchall()

tweet_count_list = [tweet_count for (_, tweet_count) in results]

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