#!/usr/bin/env python3
folder = '/home/weiss/rs_data/recsys-challenge-2021/'
all_features = ["text_ tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count", "enaging_user_is_verified",\
               "enaging_user_account_creation", "engagee_follows_engager"]

all_features_to_idx = dict(zip(all_features, range(len(all_features))))
labels_to_idx = {"reply_timestamp": 20, "retweet_timestamp": 21, "retweet_with_comment_timestamp": 22, "like_timestamp": 23}

# with open("train.tsv", encoding="utf-8") as f:
with open(folder+"part-00000", encoding="utf-8") as f:
  for line in f.readlines():
    line = line.strip()
    features = line.split("\x01")
    for feature, idx in all_features_to_idx.items():
      print("feature {} has value {}".format(feature, features[idx]))

    for label, idx in labels_to_idx.items():
      print("label {} has value {}".format(label, features[idx]))
    break
