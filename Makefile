.DEFAULT_GOAL := help
DATA ?= $(HOME)/scratch/relrel-rt
PYTHON ?= pixi run python

NAME ?=
DAG ?= global
DECODING ?= quantile
ALPHA ?= 1.0,5.0
FIXED_MLP ?=
SAMPLER ?= family
MAX_HOPS ?= 2
COL_PARENT_CAP ?= 8
DEGREE ?=
SCM_FAMILIES ?=
PARENT_POLICY ?= bounded
CELL_PARENT_CAP ?= 8
ROW_POOLING ?= mean
DATES ?= activity
EXPORT_CAT ?= plurel_bool
MAX_FKS ?= 5
COUNT ?= 128
SEED_START ?= 10000
WORKERS ?= 1
TINY ?= 0
SCHEMA_FILE ?=
MATCH_SEEDS_FROM ?=
MATCH_EDGES_FROM ?=

synth_args = \
	--data-root "$(DATA)" \
	$(if $(NAME),--name "$(NAME)") \
	--dag "$(DAG)" \
	--decoding "$(DECODING)" \
	--alpha "$(ALPHA)" \
	$(if $(FIXED_MLP),--fixed-mlp "$(FIXED_MLP)") \
	--sampler "$(SAMPLER)" \
	--max-hops "$(MAX_HOPS)" \
	--col-parent-cap "$(COL_PARENT_CAP)" \
	$(if $(DEGREE),--degree "$(DEGREE)") \
	$(if $(SCM_FAMILIES),--scm-families "$(SCM_FAMILIES)") \
	--parent-policy "$(PARENT_POLICY)" \
	--cell-parent-cap "$(CELL_PARENT_CAP)" \
	--row-pooling "$(ROW_POOLING)" \
	--dates "$(DATES)" \
	--export-cat "$(EXPORT_CAT)" \
	--max-fks "$(MAX_FKS)" \
	--count "$(COUNT)" \
	--seed-start "$(SEED_START)" \
	--workers "$(WORKERS)" \
	$(if $(filter 1,$(TINY)),--tiny) \
	$(if $(SCHEMA_FILE),--schema-file "$(SCHEMA_FILE)") \
	$(if $(MATCH_SEEDS_FROM),--match-seeds-from "$(MATCH_SEEDS_FROM)") \
	$(if $(MATCH_EDGES_FROM),--match-edges-from "$(MATCH_EDGES_FROM)")

.PHONY: help synthesize export generate
help:
	@echo 'Direct hyperparameter-driven synthesis (no SUITE/ARM):'
	@echo '  make generate NAME=local-orig   DAG=local  DECODING=original COUNT=128 WORKERS=8  # bit-exact PluRel v1.0.0'
	@echo '  make generate NAME=local-quant  DAG=local  DECODING=quantile COUNT=128 WORKERS=8'
	@echo '  make generate NAME=global-orig  DAG=global DECODING=original MAX_HOPS=2 COUNT=128 WORKERS=8'
	@echo '  make generate NAME=global-quant DAG=global DECODING=quantile MAX_HOPS=2 COUNT=128 WORKERS=8'
	@echo '  make generate NAME=global-dir   DAG=global DECODING=quantile SAMPLER=direct MATCH_EDGES_FROM=global-quant WORKERS=8'
	@echo '  make generate NAME=holdout-val  DAG=global DECODING=quantile COUNT=16 SEED_START=100000 WORKERS=8'
	@echo 'Targets: synthesize, export (requires NAME), generate (synthesize + export).'
	@echo 'Overrides: DATA, NAME, DAG (local|global), DECODING (original|quantile), ALPHA, FIXED_MLP (0|1),'
	@echo '  SAMPLER (family|direct), MAX_HOPS (1|2), COL_PARENT_CAP, DEGREE, SCM_FAMILIES, PARENT_POLICY (bounded|full),'
	@echo '  CELL_PARENT_CAP, ROW_POOLING (mean|sum), DATES (activity|none), EXPORT_CAT (plurel_bool|text|raw_int),'
	@echo '  MAX_FKS, COUNT, SEED_START, WORKERS, TINY=1, SCHEMA_FILE, MATCH_SEEDS_FROM, MATCH_EDGES_FROM.'

synthesize:
	$(PYTHON) -m relrel.experiments synthesize $(synth_args)

export:
	$(PYTHON) -m relrel.experiments export --data-root "$(DATA)" --name "$(NAME)"

generate:
	$(PYTHON) -m relrel.experiments generate $(synth_args)
