import argparse

from Dataset.google_re_T2P import GoogleReT2P
from Dataset.trex_T2P import TrexT2P
from Dataset.umls_T2P import UmlsT2P
from Dataset.wiki_bio_T2P import WikiBioT2P
from robustness_eval.get_dataset import kg_get_dataset

import os
import pickle

from options import get_args
from KGB_FSA.KGB_FSA import KGB_FSA
import nltk

"""
if __name__ == '__main__':

    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--dataset', type=str, default='google_re')
    parser1.add_argument('--triple_file_path', type=str, default='data/kg_examples/google_re')
    parser1.add_argument('--t2p_based_model', type=str, default='template')
    parser1.add_argument('--batch_size', type=int, default=8)
    args1 = parser1.parse_args()


    parser2 = argparse.ArgumentParser()
    parser2.add_argument('--dataset', type=str, default='trex')
    parser2.add_argument('--triple_file_path', type=str, default='data/kg_examples/trex')
    parser2.add_argument('--t2p_based_model', type=str, default='template')
    parser2.add_argument('--batch_size', type=int, default=8)
    args2 = parser2.parse_args()

    parser3 = argparse.ArgumentParser()
    parser3.add_argument('--dataset', type=str, default='umls')
    parser3.add_argument('--triple_file_path', type=str, default='data/kg_examples/umls/triples_processed')
    parser3.add_argument('--t2p_based_model', type=str, default='template')
    parser3.add_argument('--batch_size', type=int, default=8)
    args3 = parser3.parse_args()

    parser4 = argparse.ArgumentParser()
    parser4.add_argument('--dataset', type=str, default='wiki_bio')
    parser4.add_argument('--triple_file_path', type=str, default='data/kg_examples/wiki_bio/triples_processed')
    parser4.add_argument('--t2p_based_model', type=str, default='template')
    parser4.add_argument('--batch_size', type=int, default=8)
    args4 = parser4.parse_args()



    # gr = GoogleReT2P(args1)
    # tr = TrexT2P(args2)
    # um  = UmlsT2P(args3)
    # wb = WikiBioT2P(args4)

    # 测试内容，自主修改
    # kg_triples = wb.extract_triples()
    # print(kg_triples)
    # prompts = wb.template_based_generate_prompts(kg_triples)
    # print(prompts)

    test_loader, label_list = kg_get_dataset(args3)

    with open('test.txt', 'w', encoding='utf-8') as file:
        for batch_size_data in test_loader:
            for single_data in batch_size_data:
                values_now, label_now = single_data
                for value_now in values_now:
                    file.write(str(value_now) + '\n')
                file.write(str(label_now) + '\n')
                file.write('\n')
        for item in label_list:
            file.write(item + '\n')
"""

if __name__ == '__main__':
    args = get_args()
    for model in ['gpt-3.5-turbo-instruct', 'gpt-4-turbo', 'gpt-4o', 'gpt-4-0613']:
        args.version = model
        if args.version == 'gpt-3.5-turbo-instruct':
            args.API_base = 'https://api.openai.com/v1'
        else:
            args.API_base = 'https://api.openai.com/v1/chat'
        for dataset_now in ['google_re', 'trex', 'umls', 'wiki_bio']:
            args.dataset = dataset_now
            args.pkl_file_path = 'info/{}/{}_info.pkl'.format(args.version, args.dataset)
            # for t2p_based_model in ['template', 'llm']:
            for t2p_based_model in ['llm']:

                print(args.version)

                args.t2p_based_model = t2p_based_model
                if args.dataset == 'google_re':
                    args.triple_file_path = 'data/kg_examples/google_re'
                elif args.dataset == 'trex':
                    args.triple_file_path = 'data/kg_examples/trex'
                elif args.dataset == 'umls':
                    args.triple_file_path = 'data/kg_examples/umls/triples_processed'
                elif args.dataset == 'wiki_bio':
                    args.triple_file_path = 'data/kg_examples/wiki_bio/triples_processed'


                test_loader_now, label_list_now = kg_get_dataset(args)
                test_loader_and_label_list_now = {
                    "test_loader": test_loader_now,
                    "label_list": label_list_now
                }
                file_path = os.path.join(
                    'generated_original_prompts',
                    '{}_{}_{}.pkl'.format(args.version, args.dataset, args.t2p_based_model)
                )
                with open(file_path, 'wb') as file:
                    pickle.dump(test_loader_and_label_list_now, file)

                print(
                    "\n\n\n\n\n\n\n\n\n\nEND\nModel Version:{}  Dataset:{}  t2p_based_model:{}\n\n\n\n\n\n\n\n\n\n".format(
                        args.version, args.dataset, args.t2p_based_model
                    )
                )

"""
if __name__ == '__main__':
    # 'google_re' 'trex' 'umls' 'wiki_bio'
    dataset = 'wiki_bio'
    with open(os.path.join("info", "{}_info.pkl".format(dataset)), "rb") as f:
        # td_fsexample_info = {'fs_example': [], 'td': []}
        td_fsexample_info = pickle.load(f)
        # print(td_fsexample_info)
        fs_examples = td_fsexample_info['fs_example']
        td = td_fsexample_info['td']

    with open('{}.txt'.format(dataset), 'w', encoding='utf-8') as file:
        file.write(str(td_fsexample_info))

    with open('{}_fs_example.txt'.format(dataset), 'w', encoding='utf-8') as file:
        for fs_example in fs_examples:
            for single_example in fs_example:
                file.write(str(single_example))
                file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write('\n')
            file.write('\n')

    with open('{}_td.txt'.format(dataset), 'w', encoding='utf-8') as file:
        for item in td:
            file.write(str(item))
            file.write('\n')
"""

"""
if __name__ == '__main__':
    # nltk.download("punkt")
    args = get_args()
    for dataset_now in ['google_re', 'trex', 'umls', 'wiki_bio']:
        args.dataset = dataset_now
        if args.dataset == 'google_re':
            args.triple_file_path = 'data/kg_examples/google_re'
        elif args.dataset == 'trex':
            args.triple_file_path = 'data/kg_examples/trex'
        elif args.dataset == 'umls':
            args.triple_file_path = 'data/kg_examples/umls/triples_processed'
        elif args.dataset == 'wiki_bio':
            args.triple_file_path = 'data/kg_examples/wiki_bio/triples_processed'
        kgb_fsa = KGB_FSA(["true", "entity_error", "predicate_error"], args)
        kgb_fsa.generate_info_file()
        print("\n\n\n\n\nEND{}\n\n\n\n\n\n\n\n\n\n\n\n\n\n".format(dataset_now))
"""
