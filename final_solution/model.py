from typing import Iterable

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from natasha import (
    Doc,
    MorphVocab,
    NamesExtractor,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    NewsSyntaxParser,
    Segmenter,
)
from transformers import AutoModel, AutoTokenizer


def get_issuer_map(mapper_file_path: str) -> pd.Series:
    names_n_synonyms_df = pd.read_excel(mapper_file_path)
    names_n_synonyms_df["one_string"] = names_n_synonyms_df.iloc[:, 2:].apply(
        lambda row: " ".join(row.dropna().tolist()).lower(), axis=1
    )
    mapper = names_n_synonyms_df.set_index("issuerid")["one_string"]
    return mapper


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class Model:
    def __init__(
        self,
        model_path: str = "./models/best_model.ubj",
        mapper_file_path: str = "./data/names_n_synonyms.xlsx",
    ):
        self.classifier_model = xgb.Booster()
        self.classifier_model.load_model(model_path)
        self.mapper = get_issuer_map(mapper_file_path)
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()

        self.news_emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.news_emb)
        self.syntax_parser = NewsSyntaxParser(self.news_emb)
        self.ner_tagger = NewsNERTagger(self.news_emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        self.embedding_model = AutoModel.from_pretrained(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

    def get_company_ids(self, text: str):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)

        for span in doc.spans:
            span.normalize(self.morph_vocab)

        companies = []
        for x in {_.normal for _ in doc.spans}:
            found_ones = self.mapper[
                self.mapper.str.contains(x.lower(), regex=False)
            ].index.tolist()
            companies.extend(found_ones)
        return companies

    def get_embeddings_from_messages(self, messages: Iterable[str]) -> np.ndarray:
        # Tokenize sentences
        encoded_input = self.embedding_tokenizer(
            messages, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = encoded_input.to(self.embedding_model.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)

        # Perform pooling. In this case, average pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        proper_embeddings_numpy = sentence_embeddings.cpu().detach().numpy()
        return proper_embeddings_numpy

    def find_company_ids_from_messages(self, messages: Iterable[str]):
        found_companies = []
        for message in messages:
            found_companies.append(self.get_company_ids(message))
        return found_companies

    def prepare_features(
        self,
        company_ids: list[list[int]],
        text_embeddings: np.ndarray,
    ) -> xgb.DMatrix:
        text_embeddings_repeated = np.repeat(
            text_embeddings, [len(sublist) for sublist in company_ids], axis=0
        )
        company_ids_flattened = np.concatenate(company_ids)
        features = np.concatenate(
            [company_ids_flattened[:, np.newaxis], text_embeddings_repeated], axis=1
        )
        X = xgb.DMatrix(features)
        return X

    def forward(self, messages: Iterable[str]) -> list[list[tuple[int, float]]]:
        company_ids = self.find_company_ids_from_messages(messages)
        if not company_ids:
            return [[tuple()]]
        text_embeddings = self.get_embeddings_from_messages(messages)
        X = self.prepare_features(company_ids, text_embeddings)
        results = self.classifier_model.predict(X)
        final_results: list[list[tuple[int, float]]] = []
        current_score_index = 0
        for list_of_ids_for_message in company_ids:
            final_results.append([])
            for company_id in list_of_ids_for_message:
                final_results[-1].append(
                    (company_id, float(int(results[current_score_index]) + 1))
                )
            if len(final_results[-1]) == 0:
                final_results[-1].append(tuple())
        return final_results
