#!/usr/local/bin/python3
import random
import sys
import io
import mysql.connector
import numpy as np
from flask import Flask, redirect, request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


num_seeds = 200
num_tpn = 5
elite = 10
p_mutation = 0.7
generation = 10
offsets = {0: 40, 1: 500, 2: 20, 3: 50, 4: 10}


def generate_tpn_dose():
    conn = mysql.connector.connect(unix_socket='../../tmp/mysql/mysql.sock',
                                   user='root',
                                   password='root',
                                   host='localhost',
                                   database='tpn_dose')
    cur = conn.cursor()

    cur.execute("select id,amount,kcal,A,Na,K from tpn_dose;")
    DataList = np.zeros((53, 6), dtype=np.int16)
    for row in cur.fetchall():
        # id
        DataList[row[0]][0] = row[0]
        # amount
        DataList[row[0]][1] = row[1]
        # A
        DataList[row[0]][2] = row[3]
        # Na
        DataList[row[0]][3] = row[4]
        # kcal
        DataList[row[0]][4] = row[2]
        # K
        DataList[row[0]][5] = row[5]

    # print(DataList)
    var = cur.close
    var2 = conn.close
    return DataList


def insert_history_id(target):
    conn = mysql.connector.connect(unix_socket='../../tmp/mysql/mysql.sock',
                                   user='root',
                                   password='root',
                                   host='localhost',
                                   database='tpn_dose')
    cur = conn.cursor()

    cur.execute("INSERT INTO history_id VALUES (null,{},{},{},{},{})".format(target[0], target[3], target[1], target[2], target[4]))
    conn.commit()
    var2 = conn.close


def select_tpn(id):
    conn = mysql.connector.connect(unix_socket='../../tmp/mysql/mysql.sock',
                                   user='root',
                                   password='root',
                                   host='localhost',
                                   database='tpn_dose')
    cur = conn.cursor()
    cur.execute("select Name from tpn_dose where id = {};".format(id))
    temp = cur.fetchall()
    var = cur.close
    var2 = conn.close
    return temp


def add_list(target):
    DataList = []
    for i in range(5):
        temp = select_tpn(target[i])
        DataList.append(temp[0][0])

    print(DataList)
    return DataList


def insert_history_value(target, temp):
    # print(target[0])
    # print(target[1])
    # print(target[2])
    # print(target[3])
    # print(target[4])
    # print(temp)
    conn = mysql.connector.connect(unix_socket='../../tmp/mysql/mysql.sock',
                                   user='root',
                                   password='root',
                                   host='localhost',
                                   database='tpn_dose')
    cur = conn.cursor()

    sql = "INSERT INTO history_value (tpnID,tpn1,tpn2,tpn3,tpn4,tpn5) VALUES (%s,%s,%s,%s,%s,%s)"
    val = (temp, target[0], target[1], target[2], target[3], target[4])
    cur.execute(sql, val)
    conn.commit()
    var2 = conn.close()


def get_id():
    conn = mysql.connector.connect(unix_socket='../../tmp/mysql/mysql.sock',
                                   user='root',
                                   password='root',
                                   host='localhost',
                                   database='tpn_dose')
    cur = conn.cursor()
    cur.execute("select id from history_id order by id desc limit 1")
    temp = cur.fetchall()
    var = cur.close
    var2 = conn.close
    return temp


def generate_init_genes(num_seed, num_cities):
    Gene = np.zeros((num_seed, num_cities), dtype=np.int16)
    for i in range(num_seed):
        Gene[i, ] = random.sample(range(3, 52), k=num_cities)
    return Gene


def get_targets():
    amount = request.form['amount']
    A = request.form['A']
    Na = request.form['Na']
    kcal = request.form['kcal']
    K = request.form['K']

    # amount = 100
    # A = 10
    # Na = 150
    # kcal = 100
    # K = 75

    list = [amount, A, Na, kcal, K]
    list = [int(s) for s in list]
    return list


def python_list_add(in1, in2):
    wrk = np.array(in1) + np.array(in2)
    return wrk.tolist()


def generate_rate(Amount, A, Na, KCal, k):
    A = A / Amount
    Na = Na / Amount
    KCal = KCal / Amount
    k = k / Amount
    Amount = 1
    list = [Amount, A, Na, KCal, k]
    return list


def loss_function(x, y, z):
    lossRates = []
    for i in y:
        sumList = [0, 0, 0, 0, 0]
        for m in range(5):
            list = x[i[m]][1:]
            sumList = python_list_add(sumList, list)
        sumList = generate_rate(sumList[0], sumList[1], sumList[2], sumList[3], sumList[4])
        loss = 0
        for n in range(5):
            offset = offsets[n]
            loss = loss + 0.5*(sumList[n] - z[n])*(sumList[n] - z[n])*offset
        lossRates.append(loss)

    return lossRates


def check_value(x, y):
    sumList = [0, 0, 0, 0, 0]
    for m in range(5):
        list = x[y[m]][1:]
        sumList = python_list_add(sumList, list)
    return sumList


def generate_roulette(fitness_vec):
    total = np.sum(fitness_vec)
    roulette = np.zeros(len(fitness_vec))
    for i in range(len(fitness_vec)):
        roulette[i] = fitness_vec[i]/total
    return roulette


def roulette_choice(fitness_vec):
    roulette = generate_roulette(fitness_vec)
    choiced = np.random.choice(len(roulette), 2, replace=True, p=roulette)
    return choiced


def partial_crossover(parent1, parent2):
    num = len(parent1)
    cross_point = random.randrange(1, num - 1)
    child1 = parent1
    child2 = parent2
    for i in range(num - cross_point):
        target_index = cross_point + i

        target_value1 = parent1[target_index]
        target_value2 = parent2[target_index]
        exchange_index1 = np.where(parent1 == target_value2)
        exchange_index2 = np.where(parent2 == target_value1)

        child1[target_index] = target_value2
        child2[target_index] = target_value1
        child1[exchange_index1] = target_value1
        child2[exchange_index2] = target_value2
    return child1, child2


def partial_crossover2(parent1, parent2):
    child1 = parent1
    child2 = parent2
    for i in range(2):
        child2[np.random.randint(0, 5)] = child1[np.random.randint(0, 5)]
        child1[np.random.randint(0, 5)] = child2[np.random.randint(0, 5)]

    return child1, child2


def translocation_mutation(genes, num_mutation, p_value):
    mutated_genes = genes
    for i in range(num_mutation):
        mutation_flg = np.random.choice(2, 1, p=[1-p_value, p_value])
        if mutation_flg == 1:
            mutation_value = np.random.choice(genes[i], 2, replace = False)
            mutation_position1 = np.where(genes[i] == mutation_value[0])
            mutation_position2 = np.where(genes[i] == mutation_value[1])
            mutated_genes[i][mutation_position1] = mutation_value[1]
            mutated_genes[i][mutation_position2] = mutation_value[0]
    return mutated_genes


def translocation_mutation2(genes, num_mutation, p_value):
    mutated_genes = genes
    for i in range(num_mutation):
        mutation_flg = np.random.choice(2, 1, p=[1-p_value, p_value])
        if mutation_flg == 1:
            # mutation_value = np.random.choice(genes[np.random.randint()], 2, replace=False)
            mutated_genes[i][np.random.randint(0, 5)] = np.random.randint(0, 52)
            mutated_genes[i][np.random.randint(0, 5)] = np.random.randint(0, 52)
    return mutated_genes





app = Flask(__name__)



@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        # 目標となる成分を取得
        targets = get_targets()
        if 0 in targets:
            return redirect('http://localhost:8888/localhost/index.php?error=true', code=303)
        # 輸液データを取得
        tpn = generate_tpn_dose()
        # 履歴に目標となる成分値を挿入
        insert_history_id(targets)
        # 挿入したテーブルでの管理IDを取得
        Id = get_id()
        # 各成分の量に対する割合を計算・取得
        targets_rate = generate_rate(targets[0], targets[1], targets[2], targets[3], targets[4])
        # print(Id[0][0])

        # for l in range(5):
        #     top_indivisual = [0, 0, 0, 0, 0]
        #     max_fit = 0
        #     genes = generate_init_genes(num_seeds, num_tpn)
        #     for m in range(generation):
        #         loss = loss_function(tpn, genes, targets_rate)
        #         fitness_vec = np.reciprocal(loss)
        #         child = np.zeros(np.shape(genes))
        #         for j in range(int((num_seeds - elite) / 2)):
        #             parents_indices = roulette_choice(fitness_vec)
        #             child[2 * j], child[2 * j + 1] = partial_crossover(genes[parents_indices[0]],
        #                                                                genes[parents_indices[1]])
        #
        #         for j in range(num_seeds - elite, num_seeds):
        #             child[j] = genes[np.argsort(fitness_vec)[j]]
        #
        #         child = translocation_mutation(child, num_seeds - elite, p_mutation)
        #         if max(fitness_vec) > max_fit:
        #             top_indivisual = genes[loss.index(min(loss))]
        #
        #         genes = child
        #     tempList = add_list(top_indivisual)
        #     insert_history_value(tempList, Id[0][0])
        #
        # return redirect('http://localhost:8888/localhost/index.php?id='+str(Id[0][0]), code=303)

        for i in range(5):
            top_indivisual = [0, 0, 0, 0, 0]
            max_fit = 0
            genes = generate_init_genes(num_seeds, num_tpn)
            for m in range(generation):
                loss = loss_function(tpn, genes, targets_rate)
                fitness_vec = np.reciprocal(loss)
                child = list(range(int((num_seeds - elite) / 2) + num_seeds - elite))
                for j in range(int((num_seeds - elite) / 2)):
                    parents_indices = roulette_choice(fitness_vec)
                    child[2 * j], child[2 * j + 1] = partial_crossover2(genes[parents_indices[0]], genes[parents_indices[1]])

                for j in range(num_seeds - elite, num_seeds):
                    child[j] = genes[np.argsort(fitness_vec)[j]]

                child = translocation_mutation2(child, num_seeds - elite, p_mutation)
                if max(fitness_vec) > max_fit:
                    max_fit = max(fitness_vec)
                    print(max_fit)
                    minLossIndex = loss.index(min(loss))
                    top_indivisual = genes[minLossIndex]

                child = np.array(child)
                child = [i for i in child if type(i) != int]
                for c, name in enumerate(child):
                    child[c] = name
                # print(child)
                genes = child

            # print(top_indivisual)
            # print('ここ')
            tempList = add_list(top_indivisual)
            insert_history_value(tempList, Id[0][0])
            print(check_value(tpn, top_indivisual))

        return redirect('http://localhost:8888/localhost/index.php?id='+str(Id[0][0]), code=303)


if __name__ == '__main__':
    app.run(port=8080)

