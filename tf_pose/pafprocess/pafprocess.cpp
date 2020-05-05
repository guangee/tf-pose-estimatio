#include <iostream>
#include <algorithm>
#include <math.h>
#include "pafprocess.h"

#define PEAKS(i, j, k) peaks[k+p3*(j+p2*i)]
#define HEAT(i, j, k) heatmap[k+h3*(j+h2*i)]
#define PAF(i, j, k) pafmap[k+f3*(j+f2*i)]

using namespace std;

vector<vector<float> > subset;
vector<Peak> peak_infos_line;

int roundpaf(float v);
vector<VectorXY> get_paf_vectors(float *pafmap, const int& ch_id1, const int& ch_id2, int& f2, int& f3, Peak& peak1, Peak& peak2);
bool comp_candidate(ConnectionCandidate a, ConnectionCandidate b);
/*
(h, w, d)
*/
int process_paf(int p1, int p2, int p3, float *peaks, int h1, int h2, int h3, float *heatmap, int f1, int f2, int f3, float *pafmap) {
//    const int THRE_CNT = 4;
//    const double THRESH_PAF = 0.40;
    // 要检测18个关键点， NUM_PART = 18

    vector<Peak> peak_infos[NUM_PART];
    int peak_cnt = 0;

    //解析peaks
    // part_id : 关节点  y：高 x：宽

    for (int part_id = 0; part_id < NUM_PART; part_id ++) {
        for (int y = 0; y < p1; y ++) {
            for (int x = 0; x < p2; x ++) {
                // THRESH_HEAT = 0.05
                // (y, x, part_id) 对应于第part_id个特征图的(x, y)点

                if (PEAKS(y, x, part_id) > THRESH_HEAT) {
                    Peak info;
                    info.id = peak_cnt++;
                    info.x = x;
                    info.y = y;
                    info.score = HEAT(y, x, part_id);
                    peak_infos[part_id].push_back(info);
                }
            }
        }
    }

    peak_infos_line.clear();
    for (int part_id = 0; part_id < NUM_PART; part_id ++) {
        for (int i = 0; i < (int) peak_infos[part_id].size(); i ++) {
            peak_infos_line.push_back(peak_infos[part_id][i]);
        }
    }

    // Start to Connect
    // COCOPAIRS_SIZE = 19 ，19个肢干

    vector<Connection> connection_all[COCOPAIRS_SIZE];
    for (int pair_id = 0; pair_id < COCOPAIRS_SIZE; pair_id ++) {
        vector<ConnectionCandidate> candidates;
        // COCOPAIRS 包含的是肢干连接对
        // 比如，pair_id = 0，对应于COCOPAIRS的是{1, 2}，对应的肢干是脖子到右肩连接成的肢干
        // peak_a_list 则对应于脖子的关节点， peak_b_list对应于右肩的关节点

        vector<Peak>& peak_a_list = peak_infos[COCOPAIRS[pair_id][0]];
        vector<Peak>& peak_b_list = peak_infos[COCOPAIRS[pair_id][1]];

        if (peak_a_list.size() == 0 || peak_b_list.size() == 0) {
            continue;
        }

        // 遍历所有向量的起点A

        for (int peak_a_id = 0; peak_a_id < (int) peak_a_list.size(); peak_a_id ++) {
            // 获得向量起点a

            Peak& peak_a = peak_a_list[peak_a_id];
            // 遍历向量起点B，找到与向量起点a的最佳匹配

            for (int peak_b_id = 0; peak_b_id < (int) peak_b_list.size(); peak_b_id ++) {
                // 获得向量起点a

                Peak& peak_b = peak_b_list[peak_b_id];

                // calculate vector(direction)
                // 以a为起点，b为终点的向量XY

                VectorXY vec;
                vec.x = peak_b.x - peak_a.x;
                vec.y = peak_b.y - peak_a.y;

                // 计算向量XY的模，即向量的长度

                float norm = (float) sqrt(vec.x * vec.x + vec.y * vec.y);
                // 如果长度太小就pass 了

                if (norm < 1e-12) continue;

                // 向量XY的方向余弦和方向正弦

                vec.x = vec.x / norm;
                vec.y = vec.y / norm;

                vector<VectorXY> paf_vecs = get_paf_vectors(pafmap, COCOPAIRS_NET[pair_id][0], COCOPAIRS_NET[pair_id][1], f2, f3, peak_a, peak_b);
                float scores = 0.0f;

                // criterion 1 : score treshold count
                int criterion1 = 0;
                for (int i = 0; i < STEP_PAF; i ++) {
                    // 用两个向量的方向余弦相乘再加上方向余弦相乘，如果两个向量完全一致，则等于1

                    float score = vec.x * paf_vecs[i].x + vec.y * paf_vecs[i].y;
                    scores += score;

                    // THRESH_VECTOR_SCORE = 0.05
                    if (score > THRESH_VECTOR_SCORE) criterion1 += 1;
                }

                float criterion2 = scores / STEP_PAF + min(0.0, 0.5 * h1 / norm - 1.0);

                //THRESH_VECTOR_CNT1 = 6;
                if (criterion1 > THRESH_VECTOR_CNT1 && criterion2 > 0) {
                    ConnectionCandidate candidate;
                    candidate.idx1 = peak_a_id;
                    candidate.idx2 = peak_b_id;
                    candidate.score = criterion2;
                    candidate.etc = criterion2 + peak_a.score + peak_b.score;
                    candidates.push_back(candidate);
                }
            }
        }

        vector<Connection>& conns = connection_all[pair_id];
        sort(candidates.begin(), candidates.end(), comp_candidate);
        for (int c_id = 0; c_id < (int) candidates.size(); c_id ++) {
            ConnectionCandidate& candidate = candidates[c_id];
            bool assigned = false;
            for (int conn_id = 0; conn_id < (int) conns.size(); conn_id ++) {
                if (conns[conn_id].peak_id1 == candidate.idx1) {
                    // already assigned
                    assigned = true;
                    break;
                }
                if (assigned) break;
                if (conns[conn_id].peak_id2 == candidate.idx2) {
                    // already assigned
                    assigned = true;
                    break;
                }
                if (assigned) break;
            }
            if (assigned) continue;

            Connection conn;
            conn.peak_id1 = candidate.idx1;
            conn.peak_id2 = candidate.idx2;
            conn.score = candidate.score;
            conn.cid1 = peak_a_list[candidate.idx1].id;
            conn.cid2 = peak_b_list[candidate.idx2].id;
            conns.push_back(conn);
        }
    }

    // Generate subset
    subset.clear();
    for (int pair_id = 0; pair_id < COCOPAIRS_SIZE; pair_id ++) {
        vector<Connection>& conns = connection_all[pair_id];
        int part_id1 = COCOPAIRS[pair_id][0];
        int part_id2 = COCOPAIRS[pair_id][1];

        for (int conn_id = 0; conn_id < (int) conns.size(); conn_id ++) {
            int found = 0;
            int subset_idx1=0, subset_idx2=0;
            for (int subset_id = 0; subset_id < (int) subset.size(); subset_id ++) {
                if (subset[subset_id][part_id1] == conns[conn_id].cid1 || subset[subset_id][part_id2] == conns[conn_id].cid2) {
                    if (found == 0) subset_idx1 = subset_id;
                    if (found == 1) subset_idx2 = subset_id;
                    found += 1;
                }
            }

            if (found == 1) {
                if (subset[subset_idx1][part_id2] != conns[conn_id].cid2) {
                    subset[subset_idx1][part_id2] = conns[conn_id].cid2;
                    subset[subset_idx1][19] += 1;
                    subset[subset_idx1][18] += peak_infos_line[conns[conn_id].cid2].score + conns[conn_id].score;
                }
            } else if (found == 2) {
                int membership;
                for (int subset_id = 0; subset_id < 18; subset_id ++) {
                    if (subset[subset_idx1][subset_id] > 0 && subset[subset_idx2][subset_id] > 0) {
                        membership = 2;
                    }
                }

                if (membership == 0) {
                    for (int subset_id = 0; subset_id < 18; subset_id ++) subset[subset_idx1][subset_id] += (subset[subset_idx2][subset_id] + 1);

                    subset[subset_idx1][19] += subset[subset_idx2][19];
                    subset[subset_idx1][18] += subset[subset_idx2][18];
                    subset[subset_idx1][18] += conns[conn_id].score;
                    subset.erase(subset.begin() + subset_idx2);
                } else {
                    subset[subset_idx1][part_id2] = conns[conn_id].cid2;
                    subset[subset_idx1][19] += 1;
                    subset[subset_idx1][18] += peak_infos_line[conns[conn_id].cid2].score + conns[conn_id].score;
                }
            } else if (found == 0 && pair_id < 17) {
                vector<float> row(20);
                for (int i = 0; i < 20; i ++) row[i] = -1;
                row[part_id1] = conns[conn_id].cid1;
                row[part_id2] = conns[conn_id].cid2;
                row[19] = 2;
                row[18] = peak_infos_line[conns[conn_id].cid1].score +
                          peak_infos_line[conns[conn_id].cid2].score +
                          conns[conn_id].score;
                subset.push_back(row);
            }
        }
    }

    // delete some rows
    for (int i = subset.size() - 1; i >= 0; i --) {
        if (subset[i][19] < THRESH_PART_CNT || subset[i][18] / subset[i][19] < THRESH_HUMAN_SCORE)
            subset.erase(subset.begin() + i);
    }

    return 0;
}

int get_num_humans() {
    return subset.size();
}

int get_part_cid(int human_id, int part_id) {
    return subset[human_id][part_id];
}

float get_score(int human_id) {
    return subset[human_id][18] / subset[human_id][19];
}

int get_part_x(int cid) {
    return peak_infos_line[cid].x;
}
int get_part_y(int cid) {
    return peak_infos_line[cid].y;
}
float get_part_score(int cid) {
    return peak_infos_line[cid].score;
}

//(pafmap, COCOPAIRS_NET[pair_id][0], COCOPAIRS_NET[pair_id][1], f2, f3, peak_a, peak_b)
vector<VectorXY> get_paf_vectors(float *pafmap, const int& ch_id1, const int& ch_id2, int& f2, int& f3, Peak& peak1, Peak& peak2) {
    vector<VectorXY> paf_vectors;

    // 将向量ab等分成10段

    const float STEP_X = (peak2.x - peak1.x) / float(STEP_PAF);
    const float STEP_Y = (peak2.y - peak1.y) / float(STEP_PAF);

    for (int i = 0; i < STEP_PAF; i ++) {
        // roundpaf 四舍五入，将小数变成整数
        // 求每个分段点的坐标

        int location_x = roundpaf(peak1.x + i * STEP_X);
        int location_y = roundpaf(peak1.y + i * STEP_Y);

        // 求 (location_y, location_x, 第part_id个特征图)的方向余弦和方向正弦值

        VectorXY v;
        v.x = PAF(location_y, location_x, ch_id1);
        v.y = PAF(location_y, location_x, ch_id2);
        paf_vectors.push_back(v);
    }

    return paf_vectors;
}

int roundpaf(float v) {
    return (int) (v + 0.5);
}

bool comp_candidate(ConnectionCandidate a, ConnectionCandidate b) {
    return a.score > b.score;
}
