import pyarrow.dataset as ds
import pyarrow.flight as fl
# from pyarrow.lib import RecordBatchReader


class FlightServer(fl.FlightServerBase):

    def __init__(self, location="grpc://0.0.0.0:8815", **kwargs):
        super(FlightServer, self).__init__(location, **kwargs)

        # sample_data = ds.dataset("./sample_data", format="parquet")
        sample_data = ds.dataset("s3://molbeam/tested", format="parquet")
        self.datasets = {
            b'molbeam': sample_data,
        }

    def do_get(self, context, ticket):
        dataset = self.datasets[ticket.ticket]

        # Duplicated generator just to get the schema :(
        b = dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"])
        head = next(b)
        schema = head.schema
        print('schema done')

        batches = dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"])
        return fl.GeneratorStream(schema, batches)


def main():
    FlightServer().serve()


if __name__ == '__main__':
    main()
