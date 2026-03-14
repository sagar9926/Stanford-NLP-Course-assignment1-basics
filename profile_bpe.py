"""Profile BPE training to identify bottlenecks."""
import cProfile
import pstats
from cs336_basics.train_bpe import train_bpe

cProfile.run(
    "train_bpe('tests/fixtures/corpus.en', vocab_size=500, special_tokens=['<|endoftext|>'])",
    "bpe_profile.prof",
)

stats = pstats.Stats("bpe_profile.prof")
print("=" * 80)
print("TOP 20 BY CUMULATIVE TIME")
print("=" * 80)
stats.sort_stats("cumulative").print_stats(20)

print("=" * 80)
print("TOP 20 BY TOTAL TIME (self, excluding subcalls)")
print("=" * 80)
stats.sort_stats("tottime").print_stats(20)
