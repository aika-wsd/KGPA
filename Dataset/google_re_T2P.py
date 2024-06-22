from Dataset.T2P import T2P

class GoogleReT2P(T2P):
    def __init__(self, args):
        super().__init__(args)
        # self.args = args
        self.generated_prompts = None
        self.labels_list = None
        if self.args.t2p_based_model == "template":
            (self.generated_prompts, self.subs, self.predicates, self.objs,
             self.sub_aliases, self.obj_aliases, self.labels_list) = self.template_based_generate_prompts()
        elif self.args.t2p_based_model == "llm":
            (self.generated_prompts, self.subs, self.predicates, self.objs,
             self.sub_aliases, self.obj_aliases, self.labels_list) = self.llm_based_generate_prompts()
        else:
            print("T2P based model not supported")
            exit(1)
        self.len = len(self.labels_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.generated_prompts[idx], self.subs[idx], self.predicates[idx], self.objs[idx],
             self.sub_aliases[idx], self.obj_aliases[idx], self.labels_list[idx])

    def load_predicates(self):
        predicate_dict = {}

        if self.dataset == 'google_re':
            # 记录关系的所有字典都应该有如下的格式
            predicate_dict["/people/person/date_of_birth"] = {
                "predicate": "date of birth",  # 关系本身
                # "type": "N-1",  # 关系的类型(1-N, N-1, N-M)
                "description": "date of birth",  # 对该关系的描述
            }
            predicate_dict["/people/person/place_of_birth"] = {
                "predicate": "place of birth",
                # "type": "N-1",
                "description": "place of birth",
            }
            predicate_dict["/people/deceased_person/place_of_death"] = {
                "predicate": "place of death",
                # "type": "N-1",
                "description": "place of death",
            }

        return predicate_dict

    def template_based_generate_prompts(self):
        triples = self.extract_triples()
        # 用于存储生成的内容
        generated_prompts = []
        subs_list = []
        predicates_list = []
        objs_list = []
        sub_aliases_list = []
        obj_aliases_list = []
        labels = []
        # 将每一个三元组转化为prompt
        for triple in triples:
            subs = triple["subs"]
            predicate = triple["predicate"]
            objs = triple["objs"]
            sub_aliases = triple["sub_aliases"]
            obj_aliases = triple["obj_aliases"]
            # 就是sub relation obj的关系的表述
            prompt = "{}'s {} is {}.".format(
                subs[0], predicate["predicate"], objs[0]
            )

            # 打印出产生的prompt
            print('Model Version:{}  Dataset:{}  t2p_based_model:{}:\n{}\nLabel:{}\n\n'.format(
                self.args.version,
                self.args.dataset,
                self.args.t2p_based_model,
                prompt,
                triple["label"],
            ))

            # 添加generated_prompts
            generated_prompts.append(prompt.capitalize())
            # 添加subs
            subs_list.append(str(subs))
            # 添加relations_list
            predicates_list.append(str(predicate))
            # 添加objs_list
            objs_list.append(str(objs))
            # 添加sub_aliases_list
            sub_aliases_list.append(str(sub_aliases))
            # 添加obj_aliases_list
            obj_aliases_list.append(str(obj_aliases))
            # 添加label
            labels.append(triple["label"])

        return generated_prompts, subs_list, predicates_list, objs_list, sub_aliases_list, obj_aliases_list, labels
