.PHONY: analysis prepare_batch ingest_hf

analysis:
	$(MAKE) -C data_analysis all

prepare_batch:
	$(MAKE) -C data_analysis prepare_batch

ingest_hf:
	$(MAKE) -C data_analysis ingest_hf
