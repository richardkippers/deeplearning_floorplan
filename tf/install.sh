# Install compiler 
apt-get install protobuf-compiler

# Clone TF Models, compile proto
rm -rf models
git clone https://github.com/tensorflow/models.git
cd models
git checkout 31e86e8
git reset --hard FETCH_HEAD

protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .