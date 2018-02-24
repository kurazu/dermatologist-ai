bottleneck512:
	venv/bin/python -m cancer.bottleneck 512 512

bottleneck299:
	venv/bin/python -m cancer.bottleneck 299 299

train512:
	venv/bin/python -m cancer.train 512 512

train299:
	venv/bin/python -m cancer.train 299 299
