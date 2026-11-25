.DEFAULT_GOAL := help
DATA ?= $(HOME)/scratch/relrel-rt
PYTHON ?= pixi run python
SUITE ?= factorial
SPLIT ?= train
WORKERS ?= 1
REFERENCE ?= plurel

.PHONY: help synthesize export reference
help:
	@echo 'make synthesize DATA=/shared/relrel-rt SUITE=factorial WORKERS=8'
	@echo 'make synthesize DATA=/shared/relrel-rt SUITE=sampler WORKERS=8'
	@echo 'make synthesize DATA=/shared/relrel-rt SUITE=holdout SPLIT=val'
	@echo 'make export DATA=/shared/relrel-rt SUITE=factorial'
	@echo 'make reference DATA=/shared/relrel-rt REFERENCE=plurel COUNT=4'
	@echo 'make reference DATA=/shared/relrel-rt REFERENCE=relrel COUNT=4'
	@echo 'Optional generation overrides are COUNT, SEED_START, WORKERS and TINY=1.'

synthesize:
	$(PYTHON) -m relrel.experiments synthesize --data-root "$(DATA)" --suite "$(SUITE)" --split "$(SPLIT)" --workers "$(WORKERS)" $(if $(COUNT),--count "$(COUNT)") $(if $(SEED_START),--seed-start "$(SEED_START)") $(if $(filter 1,$(TINY)),--tiny)

export:
	$(PYTHON) -m relrel.experiments export --data-root "$(DATA)" --suite "$(SUITE)" --split "$(SPLIT)"

reference:
	$(PYTHON) -m relrel.experiments reference --data-root "$(DATA)" --reference "$(REFERENCE)" $(if $(COUNT),--count "$(COUNT)") $(if $(SEED_START),--seed-start "$(SEED_START)") $(if $(filter 1,$(TINY)),--tiny)
