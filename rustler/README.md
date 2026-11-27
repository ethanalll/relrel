# rustler - legacy PluRel v1.0.0 context sampler (deprecated here)

> **Note:** This directory is the historical PluRel v1.0.0 copy retained only
> for reference/legacy CI. For active RelRel–RT experiments (including the
> `forecast-v1` leakage fix and undated-entity `p2f` support), use
> `../relational-transformer/rustler/`.

`pre.rs` contains code to preprocess datasets.

`fly.rs` contains code for on-the-fly context sampling.

`common.rs` contains data structures exchanged between `pre.rs` and `fly.rs`
(using `rkyv`: https://github.com/rkyv/rkyv).

`lib.rs` is the entry point to rustler as a library.

`main.rs` is the entry point to rustler as a standalone executable.
