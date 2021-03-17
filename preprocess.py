import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.grounding import create_matcher_patterns, ground
from utils.graph import generate_adj_data_from_grounded_concepts as generate_extracted_graph_from_grounded_concepts
from utils.hybrid_graph import find_non_adj_cpts, generate_pk as generate_hybrid_graph_structure


input_paths = {
    'cpnet': {
        'vocab': './data/cpnet/concept.txt',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
}

output_paths = {
    'cpnet': {
        'patterns': './data/cpnet/matcher_patterns.json',
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'extracted_graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
        'hybrid_graph': {
            'non-adj-pairs-train': './data/csqa/hybrid/train_cpt_pairs_1hop_hybrid.jsonl',
            'non-adj-pairs-dev': './data/csqa/hybrid/dev_cpt_pairs_1hop_hybrid.jsonl',
            'non-adj-pairs-test': './data/csqa/hybrid/test_cpt_pairs_1hop_hybrid.jsonl',
            'hybrid-train-pk': './data/csqa/hybrid/train_cpt_pairs_1hop_hybrid.jsonl.pk',
            'hybrid-dev-pk': './data/csqa/hybrid/dev_cpt_pairs_1hop_hybrid.jsonl.pk',
            'hybrid-test-pk': './data/csqa/hybrid/test_cpt_pairs_1hop_hybrid.jsonl.pk',
        },
    },

}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'csqa'], choices=['common', 'csqa'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    args = parser.parse_args()

    routines = {
        'common': [
            {'func': create_matcher_patterns, 'args': (input_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], input_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], input_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], input_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_extracted_graph_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['train'], input_paths['cpnet']['pruned-graph'],
                                                                        input_paths['cpnet']['vocab'], output_paths['csqa']['extracted_graph']['adj-train'], args.nprocs)},
            {'func': generate_extracted_graph_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['dev'], input_paths['cpnet']['pruned-graph'],
                                                                        input_paths['cpnet']['vocab'], output_paths['csqa']['extracted_graph']['adj-dev'], args.nprocs)},
            {'func': generate_extracted_graph_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['test'], input_paths['cpnet']['pruned-graph'],
                                                                        input_paths['cpnet']['vocab'], output_paths['csqa']['extracted_graph']['adj-test'], args.nprocs)},
            {'func': find_non_adj_cpts,
             'args': (output_paths['csqa']['grounded']['train'], input_paths['cpnet']['pruned-graph'],
                      input_paths['cpnet']['vocab'], output_paths['csqa']['hybrid_graph']['non-adj-pairs-train'])},
            {'func': find_non_adj_cpts,
             'args': (output_paths['csqa']['grounded']['dev'], input_paths['cpnet']['pruned-graph'],
                      input_paths['cpnet']['vocab'], output_paths['csqa']['hybrid_graph']['non-adj-pairs-dev'])},
            {'func': find_non_adj_cpts,
             'args': (output_paths['csqa']['grounded']['test'], input_paths['cpnet']['pruned-graph'],
                      input_paths['cpnet']['vocab'], output_paths['csqa']['hybrid_graph']['non-adj-pairs-test'])},
            {'func': generate_hybrid_graph_structure,
             'args': (output_paths['csqa']['extracted_graph']['adj-train'],
                      output_paths['csqa']['hybrid_graph']['non-adj-pairs-train'], input_paths['cpnet']['vocab'],
                      output_paths['csqa']['hybrid_graph']['hybrid-train-pk'])},
            {'func': generate_hybrid_graph_structure,
             'args': (output_paths['csqa']['extracted_graph']['adj-dev'],
                      output_paths['csqa']['hybrid_graph']['non-adj-pairs-dev'], input_paths['cpnet']['vocab'],
                      output_paths['csqa']['hybrid_graph']['hybrid-dev-pk'])},
            {'func': generate_hybrid_graph_structure,
             'args': (output_paths['csqa']['extracted_graph']['adj-test'],
                      output_paths['csqa']['hybrid_graph']['non-adj-pairs-test'], input_paths['cpnet']['vocab'],
                      output_paths['csqa']['hybrid_graph']['hybrid-test-pk'])},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()