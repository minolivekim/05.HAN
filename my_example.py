import pickle
from pprint import pprint

import dgl
import networkx as nx

import numpy as np
import torch
from dgl.data.utils import _get_dgl_url, download, get_download_dir
from scipy import io as sio, sparse

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def get_intermediate_nodes(hg, metapath):
    # First hop: traverse from the source node to the intermediate node
    src_type, etype1, dst_type1 = metapath[0]
    _, etype2, dst_type2 = metapath[1]

    src_nodes, mid_nodes_1 = hg.edges(etype=etype1)
    mid_nodes_2, dst_nodes = hg.edges(etype=etype2)

    triplets = []
    for src, mid in zip(src_nodes.numpy(), mid_nodes_1.numpy()):
        for mid2, dst in zip(mid_nodes_2.numpy(), dst_nodes.numpy()):
            if mid == mid2:
                triplets.append([src, mid, dst])
    
    return triplets

def get_reverse_intermediate_nodes(hg, metapath):
    # First hop: traverse from the source node to the intermediate node
    src_type, etype1, dst_type1 = metapath[0]
    _, etype2, dst_type2 = metapath[1]

    src_nodes, mid_nodes_1 = hg.edges(etype=etype1)
    mid_nodes_2, dst_nodes = hg.edges(etype=etype2)

    dict = {}
    for src, mid in zip(src_nodes.numpy(), mid_nodes_1.numpy()):
        triplets = []
        for mid2, dst in zip(mid_nodes_2.numpy(), dst_nodes.numpy()):
            if mid == mid2:
                if len(triplets) == 0:
                    triplets = [[dst, mid, src]]
                else:
                    triplets = np.concatenate([triplets, [[dst, mid, src]]])
        if src in dict:
            a = dict[src]
            dict[src] = np.concatenate([a, triplets])
        else:
            dict[src] = triplets
    
    return dict

# 각 경로를 따라가면서 중복된 엣지를 포함하는 리스트를 생성
def find_metapath_paths(hg, metapath):
    all_paths = []
    # 모든 paper 노드에 대해 경로를 찾음
    print(range(hg.num_nodes('paper')))
    for src in range(hg.num_nodes('paper')):
        paths = [(src,)]
        for etype in metapath:
            new_paths = []
            for path in paths:
                u = path[-1]
                v_list = hg.successors(u, etype=etype).tolist()
                new_paths.extend([path + (v,) for v in v_list])
            paths = new_paths
        all_paths.extend(paths)
    return all_paths

# 메타패스 경로를 사용하여 중복된 엣지를 포함하는 그래프 생성
def create_metapath_reachable_graph_with_duplicates(hg, metapath): # metapath(0, 1) / (2, 3)
    paths = find_metapath_paths(hg, metapath)
    print("for metapath ", metapath, " all paths: ")
    print(len(paths))
    # print(paths)
    src_list = [path[0] for path in paths]
    dst_list = [path[-1] for path in paths]
    return dgl.graph((src_list, dst_list), num_nodes=hg.num_nodes('paper'))


url = "dataset/ACM3025.pkl"
data_path = get_download_dir() + "/ACM3025.pkl"
download(_get_dgl_url(url), path=data_path)

print('data_path: ', data_path)
print('url :', url) 

with open(data_path, "rb") as f:
    data = pickle.load(f)
#print(data)
## DATA index
## label: sparse 3025x3 numpy.float64
## feature: sparse 3025x1870 numpy.float64
## PAP: sparse 3025x3025 numpy.float64
## PLP: sparse 3025x3025 numpy.float64
## train_idx: array [[0, ~, 100, 1061, ~1260..., 2225]]
## val_idx: array [[200, ~]]
## test_idx: array
save_prefix = 'data/ACM_processed/'

labels, features = (
    torch.from_numpy(data["label"].todense()).long(),
    torch.from_numpy(data["feature"].todense()).float(),
)

np.save(save_prefix + 'labels.npy', labels)

num_classes = labels.shape[1]
labels = labels.nonzero()[:, 1]
print("labels")
print(labels)
#print(labels.shape)
#print(features.shape)
#print("the number of classes: ", num_classes)
#print("labels: ", labels)

remove_self_loop = False
if remove_self_loop:
    num_nodes = data["label"].shape[0]
    data["PAP"] = sparse.csr_matrix(data["PAP"] - np.eye(num_nodes))
    data["PLP"] = sparse.csr_matrix(data["PLP"] - np.eye(num_nodes))

# Adjacency matrices for meta path based neighbors
# (Mufei): I verified both of them are binary adjacency matrices with self loops
author_g = dgl.from_scipy(data["PAP"])
subject_g = dgl.from_scipy(data["PLP"])
gs = [author_g, subject_g]
#print("Adj Matrix")
#print(author_g.adjacency_matrix().to_dense())
#print(author_g.adjacency_matrix())
#print(author_g.adjacency_matrix_scipy())

#print("features: ", features)
#print(data["PAP"].todense())
#print(data["PLP"].todense())

train_idx = torch.from_numpy(data["train_idx"]).long().squeeze(0)
val_idx = torch.from_numpy(data["val_idx"]).long().squeeze(0)
test_idx = torch.from_numpy(data["test_idx"]).long().squeeze(0)
print(train_idx)

num_nodes = author_g.num_nodes()
train_mask = get_binary_mask(num_nodes, train_idx)
val_mask = get_binary_mask(num_nodes, val_idx)
test_mask = get_binary_mask(num_nodes, test_idx)
#print(train_mask)

print("dataset loaded")
pprint(
    {
        "dataset": "ACM",
        "train": train_mask.sum().item() / num_nodes,
        "val": val_mask.sum().item() / num_nodes,
        "test": test_mask.sum().item() / num_nodes,
    }
)

### def load_acm_raw(remove_self_loop):
#assert not remove_self_loop
url = "dataset/ACM.mat"
data_path = get_download_dir() + "/ACM.mat"
download(_get_dgl_url(url), path=data_path)

data = sio.loadmat(data_path)
#print(data)
for k in data:
    print(k)
p_vs_l = data["PvsL"]  # paper-field?
p_vs_a = data["PvsA"]  # paper-author
p_vs_t = data["PvsT"]  # paper-term, bag of words
p_vs_c = data["PvsC"]  # paper-conference, labels come from that

# We assign
# (1) KDD papers as class 0 (data mining),
# (2) SIGMOD and VLDB papers as class 1 (database),
# (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
conf_ids = [0, 1, 9, 10, 13]
label_ids = [0, 1, 2, 2, 1]

p_vs_c_filter = p_vs_c[:, conf_ids]
p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
p_vs_l = p_vs_l[p_selected]
p_vs_a = p_vs_a[p_selected]
p_vs_t = p_vs_t[p_selected]

p_vs_c = p_vs_c[p_selected]

hg = dgl.heterograph(
    {
        ("paper", "pa", "author"): p_vs_a.nonzero(),
        ("author", "ap", "paper"): p_vs_a.transpose().nonzero(),
        ("paper", "pf", "field"): p_vs_l.nonzero(),
        ("field", "fp", "paper"): p_vs_l.transpose().nonzero(),
    }
)
# paper / author / field - 0 / 1 / 2
# pa / ap / pf / fp - 0 / 1 / 2 / 3
hg_num = dgl.heterograph(
    {
        ("paper", 0, "author"): p_vs_a.nonzero(),
        ("author", 1, "paper"): p_vs_a.transpose().nonzero(),
        ("paper", 2, "field"): p_vs_l.nonzero(),
        ("field", 3, "paper"): p_vs_l.transpose().nonzero(),
    }
)


features = torch.FloatTensor(p_vs_t.toarray())
print("=====features=====")
print(features)

pc_p, pc_c = p_vs_c.nonzero()
print(pc_p) # Row indices of non-zero elements
print("===================") 
print(pc_c)  # Column indices of non-zero elements

labels = np.zeros(len(p_selected), dtype=np.int64)
for conf_id, label_id in zip(conf_ids, label_ids):
    labels[pc_p[pc_c == conf_id]] = label_id
labels = torch.LongTensor(labels)
print("====labels====")
print(labels)
print("====hg graph==========")
print("list of node types")
print(hg.ntypes)
print(hg_num.ntypes)
print("The list of edge types")
print(hg.etypes)
print(hg_num.etypes)
print("the list of canonical edge types")
print(hg.canonical_etypes)
print(hg_num.canonical_etypes)
print("The number of nodes - author")
print(hg.num_nodes('author'))
print(hg_num.num_nodes('author'))
print("the number of edges - pa")
print(hg.num_edges('pa'))
print(hg_num.num_edges(0))
print("a dictionary to access the node features")
print(hg.ndata)
print("a dictionary to access the edge features")
print(hg.edata)
print("the successors of the given nodes")
print(hg.successors(0, etype='pa'))
print("the predecessors of the given nodes")
print(hg.predecessors(0, etype='pa'))
for node in hg.ntypes:
    print("Nodes - type: ", node)
    print(hg.nodes(node))
etypes = hg.etypes
print(hg.edata[dgl.ETYPE])
print(hg.edata[dgl.EID])

num_classes = 3

float_mask = np.zeros(len(pc_p))
for conf_id in conf_ids:
    pc_c_mask  = pc_c == conf_id
    float_mask[pc_c_mask] = np.random.permutation(
        np.linspace(0, 1, pc_c_mask.sum())
    )
print("===train_idx====")

train_idx = np.where(float_mask <= 0.2)[0]
val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
test_idx = np.where(float_mask > 0.3)[0]

print("the number of train idx", len(train_idx))
print('val idx', len(val_idx))
print('test idx', len(test_idx))
num_nodes = hg.num_nodes("paper")
train_mask = get_binary_mask(num_nodes, train_idx)
val_mask = get_binary_mask(num_nodes, val_idx)
test_mask = get_binary_mask(num_nodes, test_idx)
#print(train_mask)
print(features.shape[1])
metapath_graph = dgl.metapath_reachable_graph(hg, ("pa", "ap"))
print(metapath_graph)
#print(metapath_graph.edges(order='eid'))

g_pap = dgl.metapath_reachable_graph(hg_num, (0, 1))
g_pfp = dgl.metapath_reachable_graph(hg_num, (2, 3))
print("Graph PAP")
print(g_pap)
#print(g_pap.edges(order='eid'))
print("==============DUB g_pap & g_pfp=================")
# 중복된 엣지를 포함한 그래프 생성
g_pap_dup_e = create_metapath_reachable_graph_with_duplicates(hg_num, (0, 1))
g_pfp_dup_e = create_metapath_reachable_graph_with_duplicates(hg_num, (2, 3))
print(g_pap_dup_e)
print(g_pfp_dup_e)
print("======================")
no_dup_elist = torch.stack(g_pfp.edges(order='eid'), dim=1)
dup_elist = torch.stack(g_pfp_dup_e.edges(order='eid'), dim=1)
print("Same edges? ", torch.equal(no_dup_elist, dup_elist))
# 원래 그래프의 엣지 목록
no_dup_elist = torch.stack(g_pap.edges(order='eid'), dim=1)

# 중복된 엣지를 포함한 그래프의 엣지 목록
dup_elist = torch.stack(g_pap_dup_e.edges(order='eid'), dim=1)

# 엣지 목록을 비교하여 다른 엣지를 찾기
set_no_dup_elist = set(map(tuple, no_dup_elist.tolist()))
set_dup_elist = set(map(tuple, dup_elist.tolist()))

# 중복된 엣지 그래프에만 있는 엣지
unique_to_dup = set_dup_elist - set_no_dup_elist
print("Edges unique to the graph with duplicate edges: ", len(unique_to_dup))
print(unique_to_dup)

metapaths = [[("paper", "pa", "author"), ("author", "ap", "paper")], [("paper", "pf", "field"), ("field", "fp", "paper")]]

metapath_nodes = [[0, 1, 0], [0, 2, 0]]
metapath_edges = []
for metapath in metapaths:
    triplets = get_intermediate_nodes(hg, metapath)
    #rev_triplets_dict = get_reverse_intermediate_nodes(hg, metapath) ## DBLP input format
    #print(rev_triplets_dict)
    metapath_edges.append(triplets)
   # print(triplets)


# build the adjacency matrix for the graph consisting of paper, author and field
# 0 for paper, 1 for author, 2 for field
print("Paper data")
#print(data["P"])

num_p_nodes = hg_num.num_nodes('paper')
num_a_nodes = hg_num.num_nodes('author')
num_f_nodes = hg_num.num_nodes('field')
dim = num_a_nodes + num_p_nodes + num_f_nodes

type_mask = np.zeros((dim), dtype=int)
type_mask[num_p_nodes:num_p_nodes+num_a_nodes] = 1
type_mask[num_a_nodes+num_f_nodes:] = 2

## Save metapath reachbl graphs
dgl.save_graphs(save_prefix + 'g_pap.dgl', [g_pap_dup_e])
dgl.save_graphs(save_prefix + 'g_pfp.dgl', [g_pfp_dup_e])

G_list = []
G_list.append(g_pap_dup_e.to_networkx())
G_list.append(g_pfp_dup_e.to_networkx())
for G, metapath in zip(G_list, metapath_nodes):
    print(G)
    nx.write_adjlist(G, save_prefix + '0/' + '-'.join(map(str, metapath)) + '.adjlist')

feat_list = [sparse.csr_matrix(features), sparse.csr_matrix(np.zeros(num_a_nodes)), sparse.csr_matrix(np.zeros(num_f_nodes))]

for i in range(len(hg.ntypes)):
    sparse.save_npz(save_prefix + 'features_{}.npz'.format(i), feat_list[i])

for i in range(len(metapaths)):
    np.save(save_prefix + '0/'+'-'.join(map(str, metapath_nodes[i])) + '_idx.npy', metapath_edges[i])

# post-processing for mini-batched training
target_idx_list = np.arange(hg_num.num_nodes('paper'))
for metapath in metapath_nodes:
    edge_metapath_idx_array = np.load(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.npy')
    target_metapaths_mapping = {}
    for target_idx in target_idx_list:
        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[edge_metapath_idx_array[:, 0] == target_idx][:, ::-1]
    out_file = open(save_prefix + '{}/'.format(0) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb')
    pickle.dump(target_metapaths_mapping, out_file)
    out_file.close()

in_file = open(save_prefix + '0/0-1-0_idx.pickle', 'rb')
idx00 = pickle.load(in_file)
print(idx00)
in_file.close()
in_file = open(save_prefix + '/0/0-2-0_idx.pickle', 'rb')
idx01 = pickle.load(in_file)
in_file.close()
print(idx01)

np.save(save_prefix + 'node_types.npy', type_mask)
np.save(save_prefix + 'labels.npy', labels)
np.savez(save_prefix + 'train_val_test_idx.npz',
         val_idx = val_idx,
         train_idx=train_idx,
         test_idx=test_idx)


'''
    hg,
    features,
    labels,
    num_classes,
    train_idx,
    val_idx,
    test_idx,
    train_mask,
    val_mask,
    test_mask,
'''