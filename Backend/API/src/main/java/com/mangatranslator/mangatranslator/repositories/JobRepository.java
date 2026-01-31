package com.mangatranslator.mangatranslator.repositories;

import com.mangatranslator.mangatranslator.models.Job;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface JobRepository extends JpaRepository<Job, String> {
}
