import os

# get user click list
def get_user_list(rating_file):
    """
    :param rating_file:input file
    :return:dict,key:userid,value:[itemid1,itemid2]
    """
    if not os.path.exists(rating_file):
        return {},{}
    linenum = 0
    user_click = {}      # 定义user_click
    user_click_time = {} # 定义user_click_time
    with open(rating_file,encoding='utf-8') as fp:
        for line in fp:
            if linenum == 0:
                linenum += 1
                continue
            item = line.strip().split(",")
            if len(item)<4:
                continue
            userid,itemid,rating,timestamp = item[0],item[1],float(item[2]),item[3]
            # [userid, itemid, rating, timestamp] = item
            if userid + "_" + itemid not in user_click_time:
                user_click_time[userid + "_" + itemid] = int(timestamp)
            if rating < 3.0:
                continue
            if userid not in user_click:
                user_click[userid] = []
            user_click[userid].append(itemid)
    return user_click,user_click_time

def get_item_info(item_file):
    """
    :param item_file:input iteminfo file
    :return:a dict,key itermid:value [title,genres]
    """
    if not os.path.exists(item_file):
        return {}
    linenum = 0
    item_info = {}
    with open(item_file,encoding='utf-8') as fp:
        for line in fp:
            if linenum == 0:
                linenum += 1
                continue
            item = line.strip().split(",")
            if len(item) < 3:
                continue
            if len(item) == 3:
                itemid, title, genres = item
                # itemid, title, genres = item[0], item[1], item[2]
            elif len(item) > 3:
                itemid = item[0]
                genres = item[-1]
                title = ",".join(item[1:-1])
            if itemid not in item_info:
                item_info[itemid] = [title,genres]
    return item_info

if __name__ == "__main__":
    user_click = get_user_list('../data/ratings.txt')
    print(len(user_click))
    print(user_click['1'])
    item_info = get_item_info('../data/movies.txt')
    print(len(item_info))
    print(item_info['1'])
