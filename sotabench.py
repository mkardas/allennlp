from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion
from tqdm import tqdm
import torch
from allennlp.data import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device

BATCH_SIZE = 64
CUDA_ID = 0


def load_model(url, batch_size=BATCH_SIZE):
    archive = load_archive(url, cuda_device=CUDA_ID)
    model = archive.model
    reader = DatasetReader.from_params(archive.config["dataset_reader"])
    iterator_params = archive.config["iterator"]
    iterator_params["batch_size"] = batch_size
    data_iterator = DataIterator.from_params(iterator_params)
    data_iterator.index_with(model.vocab)
    return model, reader, data_iterator


def evaluate(model, dataset, data_iterator, evaluator):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_iterator(dataset, num_epochs=1, shuffle=False),
                          total=data_iterator.get_num_batches(dataset)):
            batch = move_to_device(batch, CUDA_ID)
            predictions = model(**batch)
            answers = {metadata['id']: prediction
                       for metadata, prediction in zip(batch['metadata'], predictions['best_span_str'])}
            evaluator.add(answers)
            if evaluator.cache_exists:
                break


evaluator = SQuADEvaluator(
    local_root="data/nlp/squad",
    model_name="BiDAF (single)",
    paper_arxiv_id="1611.01603",
    version=SQuADVersion.V11
)

model, reader, data_iter = load_model("https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz")
dataset = reader.read(evaluator.dataset_path)

evaluate(model, dataset, data_iter, evaluator)

evaluator.save()
print(evaluator.results)
