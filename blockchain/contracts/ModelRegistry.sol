// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelRegistry {

    struct Model {
        string version;
        string modelHash;
        uint256 timestamp;
    }

    Model public currentModel;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    function updateModel(string memory _version, string memory _hash) public onlyOwner {
        currentModel = Model(_version, _hash, block.timestamp);
    }

    function getModel() public view returns (string memory, string memory, uint256) {
        return (
            currentModel.version,
            currentModel.modelHash,
            currentModel.timestamp
        );
    }
}
