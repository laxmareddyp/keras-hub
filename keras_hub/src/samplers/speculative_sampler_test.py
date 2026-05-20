from keras import ops

from keras_hub.src.samplers.speculative_sampler import SpeculativeSampler
from keras_hub.src.tests.test_case import TestCase


class SpeculativeSamplerTest(TestCase):
    def setUp(self):
        super().setUp()
        # Use a simple alphabet of lowercase characters to [0, 26).
        self.int_lookup = {i: chr(i + ord("a")) for i in range(26)}
        self.char_lookup = {v: k for k, v in self.int_lookup.items()}
        self.batch_size = 1
        self.length = 12
        self.vocab_size = len(self.int_lookup)

    def _next_fn(self, target_chars):
        """Create a next function that outputs specific characters."""
        target_ids = ops.array([self.char_lookup[c] for c in target_chars])
        max_idx = len(target_chars) - 1

        def next_fn(prompt, cache, index):
            hidden_states = ops.ones([self.batch_size, 5])
            clamped_idx = ops.minimum(ops.cast(index, "int32"), max_idx)
            target_id = ops.take(target_ids, clamped_idx)
            logits = (
                ops.one_hot(
                    ops.broadcast_to(target_id, (self.batch_size,)),
                    self.vocab_size,
                )
                * 1e9
            )
            return logits, hidden_states, cache

        return next_fn

    def join_as_string(self, x):
        x = ops.convert_to_numpy(x)
        return ["".join([self.int_lookup[i] for i in s]) for s in x]

    def test_stateless_call(self):
        target_chars = list("helloworldss")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(
            num_speculative_tokens=3,
            temperature=1.0,
        )
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        self.assertTrue(self.join_as_string(output)[0].startswith("hello"))

    def test_all_accepted(self):
        target_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        self.assertEqual(self.join_as_string(output), ["abcdefghijkl"])

    def test_partial_acceptance(self):
        target_chars = list("abcdefghijkl")
        draft_chars = list("abxdefghijkl")

        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(draft_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        output_str = self.join_as_string(output)
        self.assertTrue(output_str[0].startswith("ab"))

    def test_early_stopping(self):
        target_chars = list("hellozzzzzzzz")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            stop_token_ids=[self.char_lookup["o"]],
            draft_next=draft_next,
        )
        output_str = self.join_as_string(output)
        self.assertTrue(
            "hello" in output_str[0] or output_str[0].startswith("hell")
        )

    def test_requires_draft_next(self):
        sampler = SpeculativeSampler(num_speculative_tokens=3)
        prompt = ops.full((self.batch_size, self.length), 0)

        def dummy_next(prompt, cache, index):
            return ops.zeros((1, self.vocab_size)), ops.zeros((1, 5)), cache

        with self.assertRaises(ValueError):
            sampler(next=dummy_next, prompt=prompt, index=0)

    def test_serialization(self):
        from keras_hub.src.samplers.top_p_sampler import TopPSampler

        base = TopPSampler(p=0.9, temperature=0.8)
        sampler = SpeculativeSampler(
            num_speculative_tokens=7,
            base_sampler=base,
            seed=42,
        )
        config = sampler.get_config()
        self.assertEqual(config["num_speculative_tokens"], 7)
        self.assertEqual(config["seed"], 42)
        self.assertIsNotNone(config["base_sampler"])

        restored = SpeculativeSampler.from_config(config)
        self.assertEqual(restored.num_speculative_tokens, 7)
        self.assertEqual(restored.seed, 42)

    def test_output_shape(self):
        target_chars = list("abcdefghijkl")
        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(target_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=3)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        self.assertEqual(ops.shape(output)[0], self.batch_size)
        self.assertEqual(ops.shape(output)[1], self.length)

    def test_batched_call(self):
        batch_size = 2
        length = 8

        def batched_next(prompt, cache, index):
            hidden_states = ops.ones([batch_size, 5])
            logits = (
                ops.one_hot(
                    ops.zeros((batch_size,), dtype="int32"),
                    self.vocab_size,
                )
                * 1e9
            )
            return logits, hidden_states, cache

        sampler = SpeculativeSampler(num_speculative_tokens=2)
        prompt = ops.full((batch_size, length), self.char_lookup["z"])
        output = sampler(
            next=batched_next,
            prompt=prompt,
            index=0,
            draft_next=batched_next,
        )
        output_str = self.join_as_string(output)
        self.assertEqual(output_str, ["aaaaaaaa", "aaaaaaaa"])

    def test_short_circuit_rejection(self):
        # If token 2 is wrong, tokens 3+ should be rejected even if correct.
        target_chars = list("abcdefghijkl")
        draft_chars = list("abxdefghijkl")

        target_next = self._next_fn(target_chars)
        draft_next = self._next_fn(draft_chars)

        sampler = SpeculativeSampler(num_speculative_tokens=5, temperature=1.0)
        prompt = ops.full((self.batch_size, self.length), self.char_lookup["z"])
        output = sampler(
            next=target_next,
            prompt=prompt,
            index=0,
            draft_next=draft_next,
        )
        output_str = self.join_as_string(output)
        self.assertTrue(output_str[0].startswith("ab"))
        self.assertEqual(output_str[0][2], "c")

    def test_generic_speculative_decoding_with_causal_lm(self):
        from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
            GPT2CausalLMPreprocessor,
        )
        from keras_hub.src.models.gpt2.gpt2_tokenizer import GPT2Tokenizer
        from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
        from keras_hub.src.models.gpt2.gpt2_causal_lm import GPT2CausalLM

        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["!", "<|endoftext|>"]
        vocab = sorted(set(vocab))
        vocab = dict([(token, i) for i, token in enumerate(vocab)])

        preprocessor = GPT2CausalLMPreprocessor(
            GPT2Tokenizer(vocabulary=vocab, merges=merges),
            sequence_length=8,
        )
        vocabulary_size = preprocessor.tokenizer.vocabulary_size()

        target_backbone = GPT2Backbone(
            vocabulary_size=vocabulary_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=preprocessor.sequence_length,
        )

        assistant_backbone = GPT2Backbone(
            vocabulary_size=vocabulary_size,
            num_layers=1,
            num_heads=1,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=preprocessor.sequence_length,
        )

        target_model = GPT2CausalLM(
            preprocessor=preprocessor,
            backbone=target_backbone,
        )

        assistant_model = GPT2CausalLM(
            preprocessor=preprocessor,
            backbone=assistant_backbone,
        )

        prompt = " airplane at airport"
        # Set assistant model via property — no signature changes needed
        # in any CausalLM subclass.
        target_model.assistant_model = assistant_model
        output_greedy = target_model.generate(prompt)
        self.assertTrue(len(output_greedy) > 0)

        target_model.compile(sampler="top_k")
        output_stochastic = target_model.generate(prompt)
        self.assertTrue(len(output_stochastic) > 0)

        # Verify plain generation still works after clearing assistant.
        target_model.assistant_model = None
        target_model.compile(sampler="greedy")
        output_plain = target_model.generate(prompt)
        self.assertTrue(len(output_plain) > 0)
