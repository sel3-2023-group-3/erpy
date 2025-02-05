from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Callable, Set, Optional

from tqdm import tqdm

from erpy.framework.genome import Genome
from erpy.framework.population import Population
from erpy.framework.reproducer import ReproducerConfig, Reproducer


@dataclass
class UniqueReproducerConfig(ReproducerConfig):
    uniqueness_test: Callable[[Set, Genome, Population], bool]
    max_retries: int
    initialisation_f: Optional[Callable[[Reproducer, Population], None]] = None

    @property
    def reproducer(self) -> Type[UniqueReproducer]:
        return UniqueReproducer


class UniqueReproducer(Reproducer):
    def __init__(self, config: UniqueReproducerConfig) -> None:
        super().__init__(config=config)
        self._archive = None

    @property
    def config(self) -> UniqueReproducerConfig:
        return super().config

    def _initialise_from_checkpoint(self, population: Population) -> None:
        super().initialise_from_checkpoint(population=population)

        key = "unique-reproducer-archive"
        try:
            self._archive = population.saving_data[key]
        except KeyError:
            self._archive = set()
            population.saving_data[key] = self._archive

    def initialise_population(self, population: Population) -> None:
        assert self._archive is not None

        self._initialise_from_checkpoint(population=population)

        if self.config.initialisation_f is not None:
            self.config.initialisation_f(self, population)
        else:
            num_to_generate = population.config.population_size - \
                len(population.to_evaluate)

            for i in tqdm(range(num_to_generate), desc="[UniqueReproducer] Initialisation"):
                # Create genome
                genome_id = self.next_genome_id
                genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                                   genome_id=genome_id)

                num_retries = 0
                while not self.config.uniqueness_test(self._archive, genome,
                                                      population) and num_retries < self.config.max_retries:
                    genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                                       genome_id=genome_id)

                    num_retries += 1

                # Add genome to population
                population.genomes[genome_id] = genome

                # Initial genomes should always be evaluated
                population.to_evaluate.add(genome_id)

    def reproduce(self, population: Population) -> None:
        assert self._archive is not None

        for parent_id in tqdm(population.to_reproduce, desc="[UniqueReproducer] reproduce"):
            parent_genome = population.genomes[parent_id]

            child_id = self.next_genome_id
            child_genome = parent_genome.mutate(child_id)

            num_retries = 0
            while not self.config.uniqueness_test(self._archive, child_genome,
                                                  population) and num_retries < self.config.max_retries:
                # Continue mutating the same genome until it is unique
                child_genome.genome_id = parent_id
                child_genome = child_genome.mutate(child_id)

                num_retries += 1

            if num_retries == self.config.max_retries:
                # Generate a unique random genome if mutation fails to find one
                num_retries = 0
                while not self.config.uniqueness_test(self._archive, child_genome,
                                                      population) and num_retries < self.config.max_retries:
                    child_genome = self.config.genome_config.genome.generate(config=self.config.genome_config,
                                                                             genome_id=child_id)
                    num_retries += 1

            if num_retries < self.config.max_retries:
                # Add the unique child to the population
                population.genomes[child_genome.genome_id] = child_genome

                # New children should always be evaluated
                population.to_evaluate.add(child_genome.genome_id)

        population.logging_data["UniqueReproducer/number_of_unique_genomes"] = len(
            self._archive)

    @property
    def archive(self) -> Set:
        assert self._archive is not None
        return self._archive
