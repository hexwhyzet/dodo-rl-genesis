DOCKER_REPO  ?= ultravanish/dodo-rl-genesis
VERSION      ?= latest
DOCKER_IMAGE  = $(DOCKER_REPO):$(VERSION)

.PHONY: install assets train-local train eval tensorboard lint docker-build docker-push

install:
	uv sync

assets:
	./setup.sh

preview:
	uv run python scripts/preview_pose.py

train-local:
	uv run python scripts/dodo_train.py --cpu --viewer -B 16 --max_iterations 1000

train:
	uv run python scripts/dodo_train.py -B 4096 --max_iterations 1000

balance-local:
	uv run python scripts/dodo_balance.py --cpu --viewer -B 16 --max_iterations 1000 \
		$(if $(RESUME),--checkpoint $(shell ls -t runs/dodo-balance/*.pt 2>/dev/null | head -1),)

balance:
	uv run python scripts/dodo_balance.py -B 4096 --max_iterations 3000

balance-resume:
	uv run python scripts/dodo_balance.py -B 4096 --max_iterations 3000 \
		--checkpoint $(shell ls -t runs/dodo-balance/*.pt 2>/dev/null | head -1)

eval:
	uv run python scripts/eval.py --checkpoint $(CHECKPOINT)

tensorboard:
	uv run tensorboard --logdir runs/

lint:
	uv run python -m py_compile src/envs/dodo_env.py scripts/dodo_train.py && echo "OK"

docker-build:
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE) .
	$(if $(filter-out latest,$(VERSION)),docker tag $(DOCKER_IMAGE) $(DOCKER_REPO):latest)

docker-push:
	docker push $(DOCKER_IMAGE)
	$(if $(filter-out latest,$(VERSION)),docker push $(DOCKER_REPO):latest)
