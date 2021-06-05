import pyarrow.dataset as ds
import pyarrow.flight as fl
# from pyarrow.lib import RecordBatchReader


class FlightServer(fl.FlightServerBase):

    def __init__(self, location="grpc://0.0.0.0:8815", **kwargs):
        super(FlightServer, self).__init__(location, **kwargs)

        sample_data = ds.dataset("./sample_data", format="parquet")
        self.datasets = {
            b'molbeam': sample_data,
        }

    def do_get(self, context, ticket):
        dataset = self.datasets[ticket.ticket]
        batches = dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"])
        b = dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"])
        s = list(b)[0].schema

        return fl.GeneratorStream(s, batches)


def main():
    FlightServer().serve()


if __name__ == '__main__':
    main()
