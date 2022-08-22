import os
from pathlib import Path
from typing import List, Tuple

# from jsonargparse import parse_as_dict_patch

import numpy as np

from cdt.eval.CodeBLEU import bleu, dataflow_match, syntax_match, weighted_ngram_match
from tree_sitter import Language


class Evaluator:
    """
    Python interface for using CodeBLEU, based on calc_code_bleu.py.
    """

    def __init__(
        self,
        lang: str,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.25,
        theta: float = 0.25,
    ):
        self.lang = lang
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

        # Load keywords and tree-sitter parser
        this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
        self.keywords = [
            x.strip()
            for x in open(
                this_dir / "keywords" / f"{self.lang}.txt", "r", encoding="utf-8"
            ).readlines()
        ]
        self.parser_language = Language(
            this_dir.parent / "ts-lang" / f"{self.lang}.so", lang
        )

    @staticmethod
    def make_weights(reference_tokens, key_word_list):
        return {
            token: 1 if token in key_word_list else 0.2 for token in reference_tokens
        }

    def corpus_code_bleu(
        self, refs_toks: List[List[List[str]]], hyps_toks: List[List[str]]
    ) -> float:
        """
        Calculates CodeBLEU for the given references and hypotheses (should be tokenized).
        :param refs_toks: the references, num_item * num_ref * num_tok.
        :param hyps_toks: the hypotheses, num_item * num_tok.
        :return: corpus-level CodeBLEU score
            NOTE: not to be confused with averaged sentence-level CodeBLEU score.
        """
        assert len(refs_toks) == len(hyps_toks)

        # Group tokens (for syntax match & dataflow match)
        refs = [
            [" ".join(ref_toks) for ref_toks in reference] for reference in refs_toks
        ]
        hyps = [" ".join(hyp_toks) for hyp_toks in hyps_toks]

        # Accumulate working scores and weights
        cum_weighted_score = 0
        cum_weight = 0

        # Calculate ngram match (BLEU)
        ngram_match_score = bleu.corpus_bleu(refs_toks, hyps_toks)
        cum_weighted_score += self.alpha * ngram_match_score
        cum_weight += self.alpha

        # Calculate weighted ngram match
        refs_toks_with_weights = [
            [
                [reference_tokens, self.make_weights(reference_tokens, self.keywords)]
                for reference_tokens in reference
            ]
            for reference in refs_toks
        ]
        weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
            refs_toks_with_weights, hyps_toks
        )
        cum_weighted_score += self.beta * weighted_ngram_match_score
        cum_weight += self.beta

        # Calculate syntax match
        try:
            syntax_match_score = syntax_match.corpus_syntax_match(
                refs, hyps, self.lang, parser_language=self.parser_language
            )
        except ZeroDivisionError:
            # Syntax match not working, ignore this part
            syntax_match_score = np.NaN
            pass
        else:
            cum_weighted_score += self.gamma * syntax_match_score
            cum_weight += self.gamma

        # Calculate dataflow match
        dataflow_match_score = dataflow_match.corpus_dataflow_match(
            refs, hyps, self.lang, parser_language=self.parser_language
        )
        if dataflow_match_score is not np.NaN:
            cum_weighted_score += self.theta * dataflow_match_score
            cum_weight += self.theta
            # else, ignore this part

        # print(f"{ngram_match_score:.3f} | {weighted_ngram_match_score:.3f} | {syntax_match_score:.3f} | {dataflow_match_score:.3f} > {cum_weighted_score/cum_weight:.3f}")

        return (cum_weighted_score / cum_weight) * 100

    def sentence_code_bleu(
        self, refs_toks: List[List[str]], hyp_toks: List[str]
    ) -> float:
        """
        Calculates CodeBLEU for the given references and hypothesis (should be tokenized).
        :param refs_toks: the references, num_ref * num_tok.
        :param hyp_toks: the hypothesis, num_tok.
        :return: sentence-level CodeBLEU score.
        """
        return self.corpus_code_bleu([refs_toks], [hyp_toks])

    def code_bleu_scores_list(
        self, refs_toks: List[List[List[str]]], hyps_toks: List[List[str]]
    ) -> List[float]:
        """
        Calculate CodeBLEU for each example (references and hypothesis) and return a list of scores
        for significance test.
        :param refs_toks: the references, num_item * num_ref * num_tok.
        :param hyps_toks: the hypotheses, num_item * num_tok.
        :return: a list of sentence-level CodeBLEU score for examples;
        """
        code_bleu_scores_list = []
        for ref_toks, hyp_toks in zip(refs_toks, hyps_toks):
            code_bleu_scores_list.append(self.sentence_code_bleu(ref_toks, hyp_toks))
        return code_bleu_scores_list
