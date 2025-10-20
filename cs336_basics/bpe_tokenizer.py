import regex as re
import collections
import json
import itertools
from .utils import *
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Pair:
    def __init__(self, pair, cnt):
        self.pair = pair
        self.cnt = cnt

    def __lt__(self, other):
        if self.cnt != other.cnt:
            return self.cnt > other.cnt
        elif len(self.pair) < len(other.pair):
            return -1
        elif len(self.pair) == len(other.pair) and self.pair < other.pair:
            return -1
        else:
            return 1


class BpeTokenizer:
    def __init__(self, input_path, vocab_size, special_tokens):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.next_token_id = 0
        self.vocab = dict()
        self.merges = list()
        self.debug = 0
        self.debug_dic1 = {}
        self.debug_dic2 = {}

    def init_vocab(self):
        for token in self.special_tokens:
            self.vocab[self.next_token_id] = bytes(token, encoding='utf-8')
            self.next_token_id += 1
        for ind in range(256):
            self.vocab[self.next_token_id] = bytes([ind])
            self.next_token_id += 1

    def update_one_vocab(self, byte_list_count_list):
        byte_pair_dic = collections.defaultdict(int)
        max_pair_list = []
        max_cnt = 0
        for byte_list, v in byte_list_count_list:
            for i in range(len(byte_list)-1):
                pair_key = (byte_list[i], byte_list[i+1])
                byte_pair_dic[pair_key] += v
                max_cnt = max(byte_pair_dic[pair_key], max_cnt)

        for k, v in byte_pair_dic.items():
            if v == max_cnt:
                max_pair_list.append(k)
        max_pair = max_pair_list[0]
        for pair in max_pair_list:
            if len(max_pair) < len(pair):
                max_pair = pair
            elif len(max_pair) == len(pair) and pair > max_pair:
                max_pair = pair

        self.vocab[self.next_token_id] = max_pair[0] + max_pair[1]
        self.next_token_id += 1
        self.merges.append(max_pair)

        for jj in range(len(byte_list_count_list)):
            byte_list, v = byte_list_count_list[jj]
            flag = False
            for i in range(len(byte_list) - 1):
                if max_pair == (byte_list[i], byte_list[i+1]):
                    flag = True
            if flag:
                new_byte_list = []
                ii = 0
                while ii < len(byte_list) - 1:
                    if max_pair == (byte_list[ii], byte_list[ii+1]):
                        new_byte_list.append(byte_list[ii] + byte_list[ii+1])
                        ii += 2
                    else:
                        new_byte_list.append(byte_list[ii])
                        ii += 1
                while ii < len(byte_list):
                    new_byte_list.append(byte_list[ii])
                    ii += 1
                byte_list_count_list[jj] = (tuple(new_byte_list), v)
        pass

    def del_pair(self, index, c, byte_pair_dic, old_pair, shift):
        if old_pair is None:
            return
        row, col = index
        try:
            pair_value = byte_pair_dic.get(old_pair)
            tmp_index_dic = pair_value[1]
        except:
            print()
            print(old_pair)
            exit()
        col_list = tmp_index_dic.get(row)
        for ic in range(len(col_list)):
            if col_list[ic] == col + shift:
                del col_list[ic]
                break
        if len(col_list) == 0:
            tmp_index_dic.pop(row)
        pair_value[0] -= c
        if pair_value[0] <= 0:
            byte_pair_dic.pop(old_pair)

        pass

    def add_pair(self, index, c, byte_pair_dic, new_pair, shift):
        if new_pair is None:
            return
        row, col = index
        if new_pair in byte_pair_dic:
            pair_value = byte_pair_dic.get(new_pair)
            pair_value[0] += c
            tmp_index_dic = pair_value[1]
            if row in tmp_index_dic:
                tmp_index_dic[row].append(col+shift)
            else:
                tmp_index_dic[row] = [col+shift]
        else:
            tmp_index_dic = {}
            tmp_index_dic[row] = [col+shift]
            byte_pair_dic[new_pair] = [c, tmp_index_dic]
        pass

    # @ana_profile
    def update_all_vocab(self, byte_list_count_list):
        byte_pair_dic = {}
        for i, (byte_list, c) in enumerate(byte_list_count_list):
            for j in range(len(byte_list) - 1):
                pair_key = (byte_list[j], byte_list[j + 1])
                if pair_key in byte_pair_dic:
                    pair_value = byte_pair_dic[pair_key]
                    pair_value[0] += c
                    tmp_dic = pair_value[1]
                    if i in tmp_dic:
                        tmp_dic[i].append(j)
                    else:
                        tmp_dic[i] = [j]
                else:
                    tmp_dic = {}
                    tmp_dic[i] = [j]
                    byte_pair_dic[pair_key] = [c, tmp_dic]
        print()
        print((b'i', b'n'), byte_pair_dic[(b'i', b'n')])
        print()
        print((b'n', b'i'), byte_pair_dic[(b'n', b'i')])

        while self.next_token_id < self.vocab_size and byte_pair_dic:
            most_freq_kv = max(byte_pair_dic.items(), key=lambda x: (x[1][0], x[0]))
            most_freq_pair = most_freq_kv[0]
            self.vocab[self.next_token_id] = most_freq_pair[0] + most_freq_pair[1]
            self.next_token_id += 1
            self.merges.append(most_freq_pair)

            cnt, tmp_dic = byte_pair_dic.get(most_freq_pair)
            for row, col_list in tmp_dic.items():
                for col in reversed(sorted(col_list)):
                    byte_list, c = byte_list_count_list[row]
                    left_pair = None if col == 0 else (byte_list[col-1], byte_list[col])
                    right_pair = None if col == len(byte_list) - 2 else (byte_list[col+1], byte_list[col+2])
                    byte_list[col] += byte_list[col+1]
                    del byte_list[col+1]
                    new_left_pair = None if col == 0 else (byte_list[col - 1], byte_list[col])
                    new_right_pair = None if col == len(byte_list) - 1 else (byte_list[col], byte_list[col + 1])

                    if left_pair == (b'0', b'0'):
                        print(row, (b'0', b'0'), tmp_dic)
                        print(byte_list, left_pair, right_pair, new_left_pair, new_right_pair)

                    self.del_pair((row, col), c, byte_pair_dic, left_pair, -1)
                    self.del_pair((row, col), c, byte_pair_dic, right_pair, 1)
                    self.add_pair((row, col), c, byte_pair_dic, new_left_pair, -1)
                    self.add_pair((row, col), c, byte_pair_dic, new_right_pair, 0)
                    update_pos_set = set()
                    for ind in range(col+1, len(byte_list)-1):
                        update_pos_pair = (byte_list[ind], byte_list[ind+1])
                        if update_pos_pair in update_pos_set:
                            continue
                        update_pos_set.add(update_pos_pair)
                        pair_value = byte_pair_dic.get(update_pos_pair)
                        if row in pair_value[1]:
                            ind_list = pair_value[1].get(row)
                            for i1, j1 in enumerate(ind_list):
                                if j1 >= col + 1:
                                    ind_list[i1] -= 1

            byte_pair_dic.pop(most_freq_pair)

        pass

    # @ana_profile
    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        self.init_vocab()
        with open(self.input_path, 'r', encoding='utf-8') as fr:
            text_utf8 = fr.read()

        escaped_tokens = [re.escape(st) for st in self.special_tokens]  # 返回 "<\|endoftext\|>"
        split_pattern = "|".join(escaped_tokens)
        documents = [part for part in re.split(split_pattern, text_utf8) if part]

        word_list = list(itertools.chain(*[re.findall(PAT, doc) for doc in documents]))
        # text_list_iter = re.finditer(PAT, text_utf8)
        byte_list_count_dic = collections.defaultdict(int)
        for word in word_list:
            byte_list_count_dic[tuple([bytes([i]) for i in word.encode('utf-8')])] += 1
        byte_list_count_list = [(list(k), v) for k, v in byte_list_count_dic.items()]

        simple_version = False

        if simple_version:
            while self.vocab_size > self.next_token_id:
                self.update_one_vocab(byte_list_count_list)
        else:
            self.update_all_vocab(byte_list_count_list)

        return self.vocab, self.merges

